package gateway

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

func TestGeminiFilesResumableUploadAndAffinity(t *testing.T) {
	fresh(t)
	type seenReq struct {
		method string
		uri    string
		key    string
		cmd    string
		body   string
	}
	var (
		mu   sync.Mutex
		seen []seenReq
	)
	var up *httptest.Server
	up = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		k := r.Header.Get("x-goog-api-key")
		cmd := r.Header.Get("X-Goog-Upload-Command")
		mu.Lock()
		seen = append(seen, seenReq{r.Method, r.URL.RequestURI(), k, cmd, string(b)})
		mu.Unlock()

		switch {
		case r.Method == "POST" && r.URL.Path == "/upload/v1beta/files" && r.URL.Query().Get("upload_id") == "":
			if k == "AIza-k1" {
				w.WriteHeader(http.StatusTooManyRequests)
				io.WriteString(w, `{"error":{"code":429,"message":"quota"}}`)
				return
			}
			w.Header().Set("X-Goog-Upload-Url", up.URL+"/upload/v1beta/files?upload_id=sess-99")
			w.Header().Set("X-Goog-Upload-Status", "active")
			w.WriteHeader(http.StatusOK)
		case r.Method == "POST" && r.URL.Path == "/upload/v1beta/files" && r.URL.Query().Get("upload_id") == "sess-99":
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Goog-Upload-Status", "final")
			io.WriteString(w, `{"file":{"name":"files/f123","mimeType":"text/plain"}}`)
		case r.Method == "GET" && r.URL.Path == "/v1beta/files/f123":
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"name":"files/f123","state":"ACTIVE"}`)
		case r.Method == "GET" && r.URL.Path == "/v1beta/files/f123:download":
			w.Header().Set("Content-Type", "application/octet-stream")
			io.WriteString(w, "raw-file-bytes")
		case r.Method == "DELETE" && r.URL.Path == "/v1beta/files/f123":
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-k1",
		Keys:    []provider.KeyAccount{{Key: "AIza-k2", Name: "k2"}},
		Routing: "order",
		Gemini:  up.URL + "/v1beta",
		Models:  []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	h := s.Handler()

	// Resumable start: k1 returns 429 -> fails over to k2 -> X-Goog-Upload-Url rewritten.
	startReq := httptest.NewRequest("POST", "http://127.0.0.1:3425/upload/v1beta/files", strings.NewReader(`{"file":{"display_name":"hello.txt"}}`))
	startReq.Header.Set("Content-Type", "application/json")
	startReq.Header.Set("X-Goog-Upload-Protocol", "resumable")
	startReq.Header.Set("X-Goog-Upload-Command", "start")
	startRec := httptest.NewRecorder()
	h.ServeHTTP(startRec, startReq)
	if startRec.Code != 200 {
		t.Fatalf("upload start failed: %d %s", startRec.Code, startRec.Body.String())
	}
	rewritten := startRec.Header().Get("X-Goog-Upload-Url")
	if !strings.HasPrefix(rewritten, "http://127.0.0.1:3425/upload/v1beta/files/") {
		t.Fatalf("expected rewritten upload URL, got %q", rewritten)
	}
	u, err := url.Parse(rewritten)
	if err != nil {
		t.Fatal(err)
	}

	// Upload chunk + finalize via rewritten session URL.
	payload := bytes.Repeat([]byte("chunk-data-"), 1024)
	chunkReq := httptest.NewRequest("POST", u.Path, bytes.NewReader(payload))
	chunkReq.ContentLength = int64(len(payload))
	chunkReq.Header.Set("Content-Type", "text/plain")
	chunkReq.Header.Set("X-Goog-Upload-Offset", "0")
	chunkReq.Header.Set("X-Goog-Upload-Command", "upload, finalize")
	chunkRec := httptest.NewRecorder()
	h.ServeHTTP(chunkRec, chunkReq)
	if chunkRec.Code != 200 || !strings.Contains(chunkRec.Body.String(), `"files/f123"`) {
		t.Fatalf("chunk finalize failed: %d %s", chunkRec.Code, chunkRec.Body.String())
	}

	// Completed session token must be deleted (second upload returns 404).
	replayRec := httptest.NewRecorder()
	h.ServeHTTP(replayRec, httptest.NewRequest("POST", u.Path, strings.NewReader("again")))
	if replayRec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for completed session, got %d", replayRec.Code)
	}

	// GET metadata, GET :download, and DELETE must all route to k2.
	getRec := httptest.NewRecorder()
	h.ServeHTTP(getRec, httptest.NewRequest("GET", "/v1beta/files/f123", nil))
	if getRec.Code != 200 || !strings.Contains(getRec.Body.String(), "ACTIVE") {
		t.Fatalf("get file: %d %s", getRec.Code, getRec.Body.String())
	}

	dlRec := httptest.NewRecorder()
	h.ServeHTTP(dlRec, httptest.NewRequest("GET", "/v1beta/files/f123:download?alt=media", nil))
	if dlRec.Code != 200 || dlRec.Body.String() != "raw-file-bytes" {
		t.Fatalf("download file: %d %q", dlRec.Code, dlRec.Body.String())
	}

	delRec := httptest.NewRecorder()
	h.ServeHTTP(delRec, httptest.NewRequest("DELETE", "/v1beta/files/f123", nil))
	if delRec.Code != 200 {
		t.Fatalf("delete file: %d %s", delRec.Code, delRec.Body.String())
	}
	if _, ok := s.geminiState.lookupFile("files/f123"); ok {
		t.Fatal("expected file affinity deleted after DELETE")
	}

	mu.Lock()
	defer mu.Unlock()
	// seen: [0]=start(k1,429), [1]=start(k2), [2]=chunk(k2), [3]=get(k2), [4]=download(k2), [5]=delete(k2)
	if len(seen) != 6 {
		t.Fatalf("expected 6 upstream calls, got %d", len(seen))
	}
	for i := 1; i < 6; i++ {
		if seen[i].key != "AIza-k2" {
			t.Errorf("call %d used key %q, want AIza-k2", i, seen[i].key)
		}
	}
	if seen[2].body != string(payload) {
		t.Errorf("chunk body mismatch: got %d bytes, want %d", len(seen[2].body), len(payload))
	}
}

func TestGeminiFilesDirectUploadListAndRegister(t *testing.T) {
	fresh(t)
	var (
		mu    sync.Mutex
		paths []string
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.Method+" "+r.URL.RequestURI())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == "POST" && r.URL.Path == "/upload/v1beta/files":
			io.WriteString(w, `{"file":{"name":"files/direct1"}}`)
		case r.Method == "GET" && r.URL.Path == "/v1beta/files":
			io.WriteString(w, `{"files":[{"name":"files/direct1"}]}`)
		case r.Method == "POST" && r.URL.Path == "/v1beta/files:register":
			io.WriteString(w, `{"files":[{"name":"files/reg1"}]}`)
		case r.Method == "GET" && r.URL.Path == "/v1beta/generatedFiles":
			io.WriteString(w, `{"generatedFiles":[]}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-dir",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	h := s.Handler()

	upReq := httptest.NewRequest("POST", "/upload/v1beta/files?uploadType=media", strings.NewReader("hello-media"))
	upReq.Header.Set("Content-Type", "text/plain")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, upReq)
	if rec.Code != 200 {
		t.Fatalf("direct upload: %d %s", rec.Code, rec.Body.String())
	}
	if _, ok := s.geminiState.lookupFile("direct1"); !ok {
		t.Fatal("expected direct1 saved in file affinity")
	}

	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{"GET", "/v1beta/files?pageSize=10", ""},
		{"POST", "/v1beta/files:register", `{"uris":["gs://bucket/obj"]}`},
		{"GET", "/v1beta/generatedFiles", ""},
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))
		if rec.Code != 200 {
			t.Fatalf("%s %s: %d %s", tc.method, tc.path, rec.Code, rec.Body.String())
		}
	}

	mu.Lock()
	defer mu.Unlock()
	want := []string{
		"POST /upload/v1beta/files?uploadType=media",
		"GET /v1beta/files?pageSize=10",
		"POST /v1beta/files:register",
		"GET /v1beta/generatedFiles",
	}
	if len(paths) != len(want) {
		t.Fatalf("paths = %v, want %v", paths, want)
	}
	for i := range want {
		if paths[i] != want[i] {
			t.Errorf("path[%d] = %q, want %q", i, paths[i], want[i])
		}
	}
}

func TestGeminiFilesPathTraversalAndKeyRestrictions(t *testing.T) {
	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"files":[]}`)
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-sec",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.Save(provider.Provider{
		ID: "oai", Name: "OAI", Key: "sk-oai",
		Chat:   up.URL + "/v1",
		Models: []string{"gpt-4o"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	h := s.Handler()
	for _, bad := range []string{"/v1beta/files/..%2Fetc", "/v1beta/files/a%5Cb", "/v1beta/files/http:%2F%2Fevil"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", bad, nil))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("GET %s: got %d, want 400", bad, rec.Code)
		}
	}

	// Restricted key allowed only oai/gpt-4o must be refused 403 on Gemini Files routes.
	keys, secrets := newCaller(t, "OAIOnly")
	if _, err := access.Update("models-key", access.Change{Key: keys[0].ID, Models: []string{"oai/gpt-4o"}}); err != nil {
		t.Fatal(err)
	}
	lh := lanGuard(s.Handler())
	for _, tc := range [][2]string{
		{"GET", "/v1beta/files"},
		{"POST", "/upload/v1beta/files"},
		{"GET", "/v1beta/generatedFiles"},
	} {
		req := httptest.NewRequest(tc[0], tc[1], nil)
		req.RemoteAddr = "192.168.1.9:5000"
		req.Header.Set("Authorization", "Bearer "+secrets[0])
		rec := httptest.NewRecorder()
		lh.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("%s %s with OAI-only key: got %d, want 403", tc[0], tc[1], rec.Code)
		}
	}
}

func TestGeminiFilesCatalogOnlyProviderKeyAllowed(t *testing.T) {
	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"files":[]}`)
	}))
	defer up.Close()

	// Provider with empty Models relies on p.Available() from the catalog.
	p := provider.Provider{
		ID: "google", Name: "Google", Preset: "google", Key: "AIza-cat",
		Gemini: up.URL + "/v1beta",
	}
	if err := provider.Save(p); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive("google", up.URL+"/v1beta", []catalog.Model{{ID: "gemini-2.5-pro"}}); err != nil {
		t.Fatal(err)
	}
	if len(p.Available()) == 0 {
		t.Fatal("expected google provider to expose catalog models via Available()")
	}
	catModel := "google/gemini-2.5-pro"

	if err := provider.Save(provider.Provider{
		ID: "oai", Name: "OAI", Key: "sk-oai",
		Chat:   up.URL + "/v1",
		Models: []string{"gpt-4o"},
	}); err != nil {
		t.Fatal(err)
	}

	keys, secrets := newCaller(t, "GoogleCatOnly", "OAIOnly")
	if _, err := access.Update("models-key", access.Change{Key: keys[0].ID, Models: []string{catModel}}); err != nil {
		t.Fatal(err)
	}
	if _, err := access.Update("models-key", access.Change{Key: keys[1].ID, Models: []string{"oai/gpt-4o"}}); err != nil {
		t.Fatal(err)
	}

	lh := lanGuard(New().Handler())
	do := func(secret string) *httptest.ResponseRecorder {
		req := httptest.NewRequest("GET", "/v1beta/files", nil)
		req.RemoteAddr = "192.168.1.9:5000"
		req.Header.Set("Authorization", "Bearer "+secret)
		rec := httptest.NewRecorder()
		lh.ServeHTTP(rec, req)
		return rec
	}

	if rec := do(secrets[0]); rec.Code != http.StatusOK {
		t.Fatalf("key scoped to catalog model %q should get 200, got %d: %s", catModel, rec.Code, rec.Body.String())
	}
	if rec := do(secrets[1]); rec.Code != http.StatusForbidden {
		t.Fatalf("key scoped to other provider should get 403, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGeminiFilesForwardedProtoRequiresTrustProxy(t *testing.T) {
	fresh(t)
	var up *httptest.Server
	up = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Goog-Upload-Url", up.URL+"/upload/v1beta/files?upload_id=s1")
		w.WriteHeader(http.StatusOK)
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-fp",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	h := New().Handler()
	sendStart := func() string {
		req := httptest.NewRequest("POST", "http://127.0.0.1:3425/upload/v1beta/files", strings.NewReader(`{}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Goog-Upload-Protocol", "resumable")
		req.Header.Set("X-Goog-Upload-Command", "start")
		req.Header.Set("X-Forwarded-Proto", "https")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		return rec.Header().Get("X-Goog-Upload-Url")
	}

	t.Setenv("MAGPIE_TRUST_PROXY", "")
	if got := sendStart(); !strings.HasPrefix(got, "http://127.0.0.1:3425/upload/v1beta/files/") {
		t.Fatalf("without trustProxy, expected http:// upload URL, got %q", got)
	}
	t.Setenv("MAGPIE_TRUST_PROXY", "1")
	if got := sendStart(); !strings.HasPrefix(got, "https://127.0.0.1:3425/upload/v1beta/files/") {
		t.Fatalf("with trustProxy, expected https:// upload URL, got %q", got)
	}
}
