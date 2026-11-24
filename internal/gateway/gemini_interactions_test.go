package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

func TestGeminiInteractionsBaseResolution(t *testing.T) {
	cases := []struct {
		name string
		p    provider.Provider
		want string
	}{
		{
			name: "google preset",
			p:    provider.Provider{ID: "google", Chat: "https://generativelanguage.googleapis.com/v1beta/openai", Key: "k"},
			want: "https://generativelanguage.googleapis.com/v1beta",
		},
		{
			name: "explicit gemini base",
			p:    provider.Provider{ID: "gem", Gemini: "https://proxy.example/custom/v1beta/", Key: "k"},
			want: "https://proxy.example/custom/v1beta",
		},
		{
			name: "remote magpie",
			p:    provider.Provider{ID: "rem", Preset: provider.RemoteMagpiePreset, Chat: "http://10.0.0.2:3425/v1", Key: "k"},
			want: "http://10.0.0.2:3425/v1beta",
		},
		{
			name: "vertex excluded",
			p:    provider.Provider{ID: "vtx", Preset: provider.VertexPreset, Gemini: "https://aiplatform.googleapis.com/v1", Key: "k"},
			want: "",
		},
		{
			name: "account excluded",
			p:    provider.Provider{ID: "acc", Gemini: "https://generativelanguage.googleapis.com/v1beta", Account: &provider.Account{User: "u"}},
			want: "",
		},
		{
			name: "off excluded",
			p:    provider.Provider{ID: "off", Gemini: "https://generativelanguage.googleapis.com/v1beta", Key: "k", Off: true},
			want: "",
		},
		{
			name: "openai-only excluded",
			p:    provider.Provider{ID: "oai", Chat: "https://api.openai.com/v1", Key: "k"},
			want: "",
		},
	}
	for _, tc := range cases {
		if got := geminiInteractionsBase(tc.p); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestGeminiInteractionsCreateAndUsage(t *testing.T) {
	fresh(t)
	var (
		mu      sync.Mutex
		gotPath string
		gotKey  string
		gotBody string
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotPath = r.URL.RequestURI()
		gotKey = r.Header.Get("x-goog-api-key")
		gotBody = string(b)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"v1_int_1","model":"gemini-2.5-flash","status":"completed","outputs":[{"type":"text","text":"hello"}],"usage":{"total_input_tokens":120,"total_cached_tokens":40,"total_output_tokens":55,"total_thought_tokens":15}}`)
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-k1",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	req := httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"gem/gemini-2.5-flash","input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("X-Magpie-Provider") != "gem" || rec.Header().Get("X-Magpie-Model") != "gem/gemini-2.5-flash" {
		t.Fatalf("headers: provider=%q model=%q", rec.Header().Get("X-Magpie-Provider"), rec.Header().Get("X-Magpie-Model"))
	}
	mu.Lock()
	defer mu.Unlock()
	if gotPath != "/v1beta/interactions" {
		t.Fatalf("upstream path = %q", gotPath)
	}
	if gotKey != "AIza-k1" {
		t.Fatalf("x-goog-api-key = %q", gotKey)
	}
	if !strings.Contains(gotBody, `"model":"gemini-2.5-flash"`) {
		t.Fatalf("wire model not rewritten: %s", gotBody)
	}
	ur := lastRecord(t)
	if ur.Operation != "interactions" || ur.Input != 80 || ur.CacheRead != 40 || ur.Output != 70 || ur.Reasoning != 15 || ur.Served != "gemini-2.5-flash" {
		t.Fatalf("unexpected usage record: %+v", ur)
	}
}

func TestGeminiInteractionsSSEStreamAndUsage(t *testing.T) {
	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		events := []string{
			"event: interaction.start\ndata: {\"event_type\":\"interaction.start\",\"interaction\":{\"id\":\"v1_sse_1\",\"model\":\"gemini-2.5-pro\"}}\n\n",
			"event: content.delta\ndata: {\"event_type\":\"content.delta\",\"delta\":{\"type\":\"text\",\"text\":\"world\"}}\n\n",
			"event: interaction.complete\ndata: {\"event_type\":\"interaction.complete\",\"interaction\":{\"id\":\"v1_sse_1\",\"model\":\"gemini-2.5-pro\",\"usage\":{\"total_input_tokens\":200,\"total_cached_tokens\":50,\"total_output_tokens\":80,\"total_thought_tokens\":20}}}\n\n",
		}
		for _, ev := range events {
			io.WriteString(w, ev)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-sse",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-pro"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	req := httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"gem/gemini-2.5-pro","input":"hi","stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "content.delta") {
		t.Fatalf("SSE failed: %d %s", rec.Code, rec.Body.String())
	}
	ur := lastRecord(t)
	if ur.Operation != "interactions" || ur.Input != 150 || ur.CacheRead != 50 || ur.Output != 100 || ur.Reasoning != 20 {
		t.Fatalf("unexpected SSE usage: %+v", ur)
	}
	owner, ok := s.geminiState.lookupInteraction("v1_sse_1")
	if !ok || owner.providerID != "gem" {
		t.Fatalf("expected interaction affinity saved, got ok=%v owner=%+v", ok, owner)
	}
}

func TestGeminiInteractionsGroupAndMultiKeyFailover(t *testing.T) {
	fresh(t)
	var (
		mu   sync.Mutex
		keys []string
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		k := r.Header.Get("x-goog-api-key")
		mu.Lock()
		keys = append(keys, k)
		mu.Unlock()
		if k == "AIza-bad" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"error":{"code":429,"message":"rate limited"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"v1_failover","model":"gemini-2.5-flash","status":"completed"}`)
	}))
	defer up.Close()

	// OpenAI-only provider in the group must be skipped for /v1beta/interactions.
	if err := provider.Save(provider.Provider{
		ID: "oai", Name: "OAI", Key: "sk-oai",
		Chat:   up.URL + "/v1",
		Models: []string{"gpt-4o"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-bad",
		Keys:    []provider.KeyAccount{{Key: "AIza-good", Name: "k2"}},
		Routing: "order",
		Gemini:  up.URL + "/v1beta",
		Models:  []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		ID:      "mix",
		Name:    "Mix",
		Routing: "order",
		Members: []string{"oai/gpt-4o", "gem/gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	req := httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"group/mix","input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200 after failover, got %d: %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(keys) != 2 || keys[0] != "AIza-bad" || keys[1] != "AIza-good" {
		t.Fatalf("expected failover from AIza-bad to AIza-good, got %v", keys)
	}
	if rec.Header().Get("X-Magpie-Provider") != "gem" {
		t.Fatalf("X-Magpie-Provider = %q", rec.Header().Get("X-Magpie-Provider"))
	}
}

func TestGeminiInteractionsAgentRouting(t *testing.T) {
	fresh(t)
	var (
		mu      sync.Mutex
		gotBody string
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotBody = string(b)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"v1_agent_1","status":"in_progress"}`)
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-agent",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}

	s := New()
	req := httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"agent":"gem/deep-research-pro-preview-12-2025","input":"research","background":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("agent interaction failed: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if !strings.Contains(gotBody, `"agent":"deep-research-pro-preview-12-2025"`) {
		t.Fatalf("expected bare agent name upstream, got %s", gotBody)
	}
}

func TestGeminiInteractionsAffinityAndFollowups(t *testing.T) {
	fresh(t)
	type seenReq struct {
		method string
		uri    string
		key    string
	}
	var (
		mu   sync.Mutex
		seen []seenReq
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		k := r.Header.Get("x-goog-api-key")
		mu.Lock()
		seen = append(seen, seenReq{r.Method, r.URL.RequestURI(), k})
		mu.Unlock()
		if r.Method == "POST" && r.URL.Path == "/v1beta/interactions" && k == "AIza-k1" {
			// Force initial creation onto k2.
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, `{"error":{"code":500,"message":"boom"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"v1_aff_k2","model":"gemini-2.5-flash","status":"completed"}`)
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

	// Create -> fails over to k2 and saves affinity for v1_aff_k2 -> k2.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"gem/gemini-2.5-flash","input":"turn 1"}`)))
	if rec.Code != 200 {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}

	// Follow-up create with previous_interaction_id should go directly to k2 (no k1 attempt).
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"gem/gemini-2.5-flash","previous_interaction_id":"v1_aff_k2","input":"turn 2"}`)))
	if rec.Code != 200 {
		t.Fatalf("turn 2: %d %s", rec.Code, rec.Body.String())
	}

	// GET with query params (?key= stripped, stream & last_event_id preserved).
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1beta/interactions/v1_aff_k2?stream=true&last_event_id=ev1&key=magpie", nil))
	if rec.Code != 200 {
		t.Fatalf("get: %d %s", rec.Code, rec.Body.String())
	}

	// POST cancel.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/interactions/v1_aff_k2/cancel", nil))
	if rec.Code != 200 {
		t.Fatalf("cancel: %d %s", rec.Code, rec.Body.String())
	}

	// DELETE evicts affinity.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("DELETE", "/v1beta/interactions/v1_aff_k2", nil))
	if rec.Code != 200 {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if _, ok := s.geminiState.lookupInteraction("v1_aff_k2"); ok {
		t.Fatal("expected affinity to be deleted after DELETE")
	}

	// Path traversal rejected.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1beta/interactions/..%2Fsecret", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for traversal id, got %d", rec.Code)
	}

	mu.Lock()
	defer mu.Unlock()
	// seen: [0]=k1(500), [1]=k2(create), [2]=k2(prev_id), [3]=k2(GET), [4]=k2(cancel), [5]=k2(DELETE)
	if len(seen) != 6 {
		t.Fatalf("expected 6 upstream requests, got %d: %+v", len(seen), seen)
	}
	for i := 1; i < 6; i++ {
		if seen[i].key != "AIza-k2" {
			t.Errorf("req %d used key %q, want AIza-k2 (%+v)", i, seen[i].key, seen[i])
		}
	}
	if seen[3].uri != "/v1beta/interactions/v1_aff_k2?stream=true&last_event_id=ev1" {
		t.Errorf("GET uri = %q", seen[3].uri)
	}
	if seen[4].uri != "/v1beta/interactions/v1_aff_k2/cancel" {
		t.Errorf("cancel uri = %q", seen[4].uri)
	}
}

func TestGeminiInteractionsKeyRestrictions(t *testing.T) {
	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"v1_restricted","model":"gemini-2.5-flash","status":"completed"}`)
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-r1",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash", "gemini-2.5-pro"},
	}); err != nil {
		t.Fatal(err)
	}

	keys, secrets := newCaller(t, "FlashOnly")
	if _, err := access.Update("models-key", access.Change{Key: keys[0].ID, Models: []string{"gem/gemini-2.5-flash"}}); err != nil {
		t.Fatal(err)
	}
	secret := secrets[0]

	s := New()
	h := lanGuard(s.Handler())

	do := func(method, path, body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, path, strings.NewReader(body))
		r.RemoteAddr = "192.168.1.5:5000"
		r.Header.Set("Authorization", "Bearer "+secret)
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w
	}

	if w := do("POST", "/v1beta/interactions", `{"model":"gem/gemini-2.5-pro","input":"nope"}`); w.Code != http.StatusForbidden {
		t.Fatalf("disallowed model should be 403, got %d: %s", w.Code, w.Body.String())
	}
	if w := do("POST", "/v1beta/interactions", `{"model":"gem/gemini-2.5-flash","input":"ok"}`); w.Code != 200 {
		t.Fatalf("allowed model should be 200, got %d: %s", w.Code, w.Body.String())
	}
	// Seed an interaction owned by gemini-2.5-pro; FlashOnly key must be refused GET on it.
	s.geminiState.saveInteraction("v1_pro_only", "gem", keyMark(provider.Provider{Key: "AIza-r1"}), "gemini-2.5-pro")
	if w := do("GET", "/v1beta/interactions/v1_pro_only", ""); w.Code != http.StatusForbidden {
		t.Fatalf("disallowed interaction GET should be 403, got %d: %s", w.Code, w.Body.String())
	}
}

func TestGeminiInteractionsPrivacyMasking(t *testing.T) {
	fresh(t)
	const secretVal = "Mv4kestrelPass28"
	placeholderRe := regexp.MustCompile(`\{\{SECRET_[a-z2-7]{8}\}\}`)
	var (
		mu      sync.Mutex
		gotBody string
	)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotBody = string(b)
		mu.Unlock()
		ph := placeholderRe.FindString(string(b))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "v1_masked",
			"model":   "gemini-2.5-flash",
			"status":  "completed",
			"outputs": []any{map[string]any{"type": "text", "text": "echo " + ph}},
		})
	}))
	defer up.Close()

	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "AIza-mask",
		Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-2.5-flash"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := settings.Save(settings.Settings{Redact: true, RedactPersonal: true}); err != nil {
		t.Fatal(err)
	}

	s := New()
	req := httptest.NewRequest("POST", "/v1beta/interactions", strings.NewReader(`{"model":"gem/gemini-2.5-flash","input":"DB_PASSWORD=`+secretVal+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if strings.Contains(gotBody, secretVal) {
		t.Fatalf("secret leaked upstream: %s", gotBody)
	}
	if !placeholderRe.MatchString(gotBody) {
		t.Fatalf("expected placeholder upstream, got %s", gotBody)
	}
	if !strings.Contains(rec.Body.String(), secretVal) {
		t.Fatalf("expected secret restored in client response, got %s", rec.Body.String())
	}
}
