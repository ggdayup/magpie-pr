package gateway

import (
	"bytes"
	"cmp"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/usage"
)

func uploadCommandHas(headerVal, cmd string) bool {
	for _, part := range strings.Split(headerVal, ",") {
		if strings.EqualFold(strings.TrimSpace(part), cmd) {
			return true
		}
	}
	return false
}

func geminiClientBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil || (trustProxy() && strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")) {
		scheme = "https"
	}
	host := r.Host
	if host == "" {
		host = Addr()
	}
	return scheme + "://" + host
}

// validGeminiFileName rejects empty names, path traversal segments, backslashes,
// schemes, and control bytes in /v1beta/files/{name...}.
func validGeminiFileName(name string) bool {
	if name == "" || strings.HasPrefix(name, "/") || strings.ContainsAny(name, "\\\x00\r\n") || strings.Contains(name, "://") {
		return false
	}
	check := func(s string) bool {
		if s == "" || strings.HasPrefix(s, "/") || strings.ContainsAny(s, "\\\x00\r\n") || strings.Contains(s, "://") {
			return false
		}
		for _, seg := range strings.Split(s, "/") {
			if seg == "." || seg == ".." {
				return false
			}
		}
		return true
	}
	if !check(name) {
		return false
	}
	if u, err := url.PathUnescape(name); err != nil || !check(u) {
		return false
	}
	return true
}

func sniffGeminiFileName(raw []byte) string {
	if !gjson.ValidBytes(raw) {
		return ""
	}
	root := gjson.ParseBytes(raw)
	return cmp.Or(root.Get("file.name").String(), root.Get("name").String())
}

func streamAndCapturePrefix(w http.ResponseWriter, src io.Reader, capBytes int) []byte {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32<<10)
	var capBuf bytes.Buffer
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if capBuf.Len() < capBytes {
				rem := capBytes - capBuf.Len()
				chunk := buf[:n]
				if len(chunk) > rem {
					chunk = chunk[:rem]
				}
				capBuf.Write(chunk)
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				break
			}
			rc.Flush()
		}
		if err != nil {
			break
		}
	}
	return capBuf.Bytes()
}

func (s *Server) geminiFilesUploadStart(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	who := callerOf(r)
	call := Call{Time: start, From: provider.Gemini, To: provider.Gemini, Agent: who.agent, Via: who.via, headers: headerLogOf(r.Context())}
	usage.Saw(agentOf(r))
	fail := func(code int, msg string) {
		call.Status, call.Error, call.Millis = code, msg, time.Since(start).Milliseconds()
		writeError(w, provider.Gemini, code, msg)
		s.record(call)
	}

	cands, keyRefused, errMsg := s.eligibleGeminiCandidates(r, r.Header.Get("X-Magpie-Provider"))
	if keyRefused {
		fail(http.StatusForbidden, errMsg)
		return
	}
	if len(cands) == 0 {
		fail(http.StatusNotFound, "no enabled Gemini Developer API provider is configured for /upload/v1beta/files")
		return
	}

	proto := r.Header.Get("X-Goog-Upload-Protocol")
	cmd := r.Header.Get("X-Goog-Upload-Command")
	isResumableStart := strings.EqualFold(proto, "resumable") && uploadCommandHas(cmd, "start")

	// Resumable start without inline upload bytes has a small metadata body, so
	// we can buffer it and fail over across keys/providers.
	if isResumableStart && !uploadCommandHas(cmd, "upload") {
		body, ok := s.requestBody(w, r, provider.Gemini)
		if !ok {
			return
		}
		for i, c := range cands {
			target := geminiUploadRoot(c.p) + "/upload/v1beta/files"
			if q := nativeQuery(r.URL.RawQuery); q != "" {
				target += "?" + q
			}
			ctx := s.metered(c.p.Via(r.Context()), c.p, "")
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
			if err != nil {
				fail(http.StatusBadGateway, err.Error())
				return
			}
			copyGeminiRequestHeaders(req.Header, r.Header)
			stampUA(ctx, c.p, req.Header)
			if err := c.p.Sign(ctx, req, provider.Gemini, body); err != nil {
				if i+1 < len(cands) && r.Context().Err() == nil {
					continue
				}
				fail(http.StatusBadGateway, err.Error())
				return
			}
			res, err := c.p.Do(s.client, req)
			if err != nil {
				if i+1 < len(cands) && r.Context().Err() == nil {
					continue
				}
				fail(http.StatusBadGateway, fmt.Sprintf("%s didn't answer: %v", c.p.Name, err))
				return
			}
			if res.StatusCode >= 400 {
				errBody, _ := io.ReadAll(io.LimitReader(res.Body, geminiErrBodyCap))
				res.Body.Close()
				if i+1 < len(cands) && r.Context().Err() == nil && retryable(res.StatusCode, errBody) {
					continue
				}
				call.Provider = c.p.ID
				call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
				call.Error = fmt.Sprintf("%s answered %d", c.p.Name, res.StatusCode)
				s.record(call)
				appendUsage(r, usage.Record{
					Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
					Provider: c.p.ID, Host: c.p.Where(),
					ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
					ProviderAccount: accountOf(c.p),
					Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
				})
				copyGeminiResponseHeaders(w.Header(), res)
				w.WriteHeader(res.StatusCode)
				w.Write(errBody)
				return
			}

			copyGeminiResponseHeaders(w.Header(), res)
			if upURL := res.Header.Get("X-Goog-Upload-Url"); upURL != "" {
				if token := s.geminiState.saveUpload(upURL, c.p.ID, keyMark(c.p)); token != "" {
					w.Header().Set("X-Goog-Upload-Url", geminiClientBaseURL(r)+"/upload/v1beta/files/"+token)
				}
			}
			w.WriteHeader(res.StatusCode)
			snippet := streamAndCapturePrefix(w, res.Body, geminiFileSniffCap)
			res.Body.Close()
			if name := sniffGeminiFileName(snippet); name != "" {
				s.geminiState.saveFile(name, c.p.ID, keyMark(c.p))
			}

			call.Provider = c.p.ID
			call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
			s.record(call)
			appendUsage(r, usage.Record{
				Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
				Provider: c.p.ID, Host: c.p.Where(),
				ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
				ProviderAccount: accountOf(c.p),
				Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
			})
			return
		}
		return
	}

	// Direct multipart/media or single-shot resumable upload: stream r.Body
	// directly without buffering the payload in memory.
	c := cands[0]
	target := geminiUploadRoot(c.p) + "/upload/v1beta/files"
	if q := nativeQuery(r.URL.RawQuery); q != "" {
		target += "?" + q
	}
	ctx := s.metered(c.p.Via(r.Context()), c.p, "")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, r.Body)
	if err != nil {
		fail(http.StatusBadGateway, err.Error())
		return
	}
	req.ContentLength = r.ContentLength
	copyGeminiRequestHeaders(req.Header, r.Header)
	stampUA(ctx, c.p, req.Header)
	if err := c.p.Sign(ctx, req, provider.Gemini, nil); err != nil {
		fail(http.StatusBadGateway, err.Error())
		return
	}
	res, err := c.p.Do(s.client, req)
	if err != nil {
		fail(http.StatusBadGateway, fmt.Sprintf("%s didn't answer: %v", c.p.Name, err))
		return
	}
	defer res.Body.Close()

	copyGeminiResponseHeaders(w.Header(), res)
	if res.StatusCode < 400 {
		if upURL := res.Header.Get("X-Goog-Upload-Url"); upURL != "" {
			if token := s.geminiState.saveUpload(upURL, c.p.ID, keyMark(c.p)); token != "" {
				w.Header().Set("X-Goog-Upload-Url", geminiClientBaseURL(r)+"/upload/v1beta/files/"+token)
			}
		}
	}
	w.WriteHeader(res.StatusCode)
	snippet := streamAndCapturePrefix(w, res.Body, geminiFileSniffCap)
	if res.StatusCode < 400 {
		if name := sniffGeminiFileName(snippet); name != "" {
			s.geminiState.saveFile(name, c.p.ID, keyMark(c.p))
		}
	}

	call.Provider = c.p.ID
	call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
	if res.StatusCode >= 400 {
		call.Error = fmt.Sprintf("%s answered %d", c.p.Name, res.StatusCode)
	}
	s.record(call)
	appendUsage(r, usage.Record{
		Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
		Provider: c.p.ID, Host: c.p.Where(),
		ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
		ProviderAccount: accountOf(c.p),
		Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
	})
}

func (s *Server) geminiFilesUploadChunk(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	who := callerOf(r)
	call := Call{Time: start, From: provider.Gemini, To: provider.Gemini, Agent: who.agent, Via: who.via, headers: headerLogOf(r.Context())}
	usage.Saw(agentOf(r))
	fail := func(code int, msg string) {
		call.Status, call.Error, call.Millis = code, msg, time.Since(start).Milliseconds()
		writeError(w, provider.Gemini, code, msg)
		s.record(call)
	}

	token := r.PathValue("session")
	sess, ok := s.geminiState.lookupUpload(token)
	if !ok {
		fail(http.StatusNotFound, "unknown or expired upload session")
		return
	}
	found, err := provider.Find(sess.providerID)
	if err != nil {
		fail(http.StatusNotFound, fmt.Sprintf("provider %q for upload session is no longer available", sess.providerID))
		return
	}
	q, ok := geminiMarkedKey(*found, sess.keyMark)
	if !ok {
		fail(http.StatusNotFound, fmt.Sprintf("key for upload session on %s is no longer available", found.Name))
		return
	}
	if accWho, held := accountHolds(r); held && !accountAllowed(accWho, candidateOf(q, "")) {
		fail(http.StatusForbidden, keyAccountsError(accWho, "Gemini Files"))
		return
	}
	if keyWho, held := keyHolds(r); held && !geminiProviderAllowedByKey(keyWho, q) {
		fail(http.StatusForbidden, keyModelError(keyWho, "Gemini Files"))
		return
	}

	ctx := s.metered(q.Via(r.Context()), q, "")
	req, err := http.NewRequestWithContext(ctx, r.Method, sess.upstreamURL, r.Body)
	if err != nil {
		fail(http.StatusBadGateway, err.Error())
		return
	}
	req.ContentLength = r.ContentLength
	copyGeminiRequestHeaders(req.Header, r.Header)
	stampUA(ctx, q, req.Header)
	if err := q.Sign(ctx, req, provider.Gemini, nil); err != nil {
		fail(http.StatusBadGateway, err.Error())
		return
	}
	res, err := q.Do(s.client, req)
	if err != nil {
		fail(http.StatusBadGateway, fmt.Sprintf("%s didn't answer: %v", q.Name, err))
		return
	}
	defer res.Body.Close()

	copyGeminiResponseHeaders(w.Header(), res)
	w.WriteHeader(res.StatusCode)
	snippet := streamAndCapturePrefix(w, res.Body, geminiFileSniffCap)

	cmd := r.Header.Get("X-Goog-Upload-Command")
	finalized := uploadCommandHas(cmd, "finalize") || strings.EqualFold(res.Header.Get("X-Goog-Upload-Status"), "final")
	if res.StatusCode < 400 {
		if finalized {
			if name := sniffGeminiFileName(snippet); name != "" {
				s.geminiState.saveFile(name, q.ID, sess.keyMark)
			}
			s.geminiState.deleteUpload(token)
		} else if uploadCommandHas(cmd, "cancel") {
			s.geminiState.deleteUpload(token)
		}
	}

	call.Provider = q.ID
	call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
	if res.StatusCode >= 400 {
		call.Error = fmt.Sprintf("%s answered %d", q.Name, res.StatusCode)
	}
	s.record(call)
	appendUsage(r, usage.Record{
		Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
		Provider: q.ID, Host: q.Where(),
		ProviderKeyID: provider.KeyID(q.Key), ProviderKeyName: q.KeyName,
		ProviderAccount: accountOf(q),
		Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
	})
}

func (s *Server) geminiFilesList(w http.ResponseWriter, r *http.Request) {
	s.geminiFilesProxy(w, r, http.MethodGet, "/files", "")
}

func (s *Server) geminiGeneratedFilesList(w http.ResponseWriter, r *http.Request) {
	s.geminiFilesProxy(w, r, http.MethodGet, "/generatedFiles", "")
}

func (s *Server) geminiFilesRegister(w http.ResponseWriter, r *http.Request) {
	s.geminiFilesProxy(w, r, http.MethodPost, "/files:register", "")
}

func (s *Server) geminiFilesGet(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !validGeminiFileName(name) {
		writeError(w, provider.Gemini, http.StatusBadRequest, "invalid file name")
		return
	}
	s.geminiFilesProxy(w, r, http.MethodGet, "/files/"+name, name)
}

func (s *Server) geminiFilesDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !validGeminiFileName(name) {
		writeError(w, provider.Gemini, http.StatusBadRequest, "invalid file name")
		return
	}
	s.geminiFilesProxy(w, r, http.MethodDelete, "/files/"+name, name)
}

func (s *Server) geminiFilesProxy(w http.ResponseWriter, r *http.Request, method, relPath, fileName string) {
	start := time.Now()
	who := callerOf(r)
	call := Call{Time: start, From: provider.Gemini, To: provider.Gemini, Agent: who.agent, Via: who.via, headers: headerLogOf(r.Context())}
	usage.Saw(agentOf(r))
	fail := func(code int, msg string) {
		call.Status, call.Error, call.Millis = code, msg, time.Since(start).Milliseconds()
		writeError(w, provider.Gemini, code, msg)
		s.record(call)
	}

	var body []byte
	if method == http.MethodPost {
		var ok bool
		body, ok = s.requestBody(w, r, provider.Gemini)
		if !ok {
			return
		}
	}

	cands, code, errMsg := s.geminiFileTargets(r, fileName)
	if code != 0 {
		fail(code, errMsg)
		return
	}

	for i, c := range cands {
		target := geminiInteractionsBase(c.p) + relPath
		if q := nativeQuery(r.URL.RawQuery); q != "" {
			target += "?" + q
		}
		ctx := s.metered(c.p.Via(r.Context()), c.p, "")
		var bodyReader io.Reader
		if len(body) > 0 {
			bodyReader = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, target, bodyReader)
		if err != nil {
			fail(http.StatusBadGateway, err.Error())
			return
		}
		copyGeminiRequestHeaders(req.Header, r.Header)
		if len(body) > 0 && req.Header.Get("Content-Type") == "" {
			req.Header.Set("Content-Type", "application/json")
		}
		stampUA(ctx, c.p, req.Header)
		if err := c.p.Sign(ctx, req, provider.Gemini, body); err != nil {
			if i+1 < len(cands) && r.Context().Err() == nil {
				continue
			}
			fail(http.StatusBadGateway, err.Error())
			return
		}

		res, err := c.p.Do(s.client, req)
		if err != nil {
			if i+1 < len(cands) && r.Context().Err() == nil {
				continue
			}
			fail(http.StatusBadGateway, fmt.Sprintf("%s didn't answer: %v", c.p.Name, err))
			return
		}

		if res.StatusCode >= 400 {
			errBody, _ := io.ReadAll(io.LimitReader(res.Body, geminiErrBodyCap))
			res.Body.Close()
			if i+1 < len(cands) && r.Context().Err() == nil && (res.StatusCode == http.StatusNotFound || retryable(res.StatusCode, errBody)) {
				continue
			}
			call.Provider = c.p.ID
			call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
			call.Error = fmt.Sprintf("%s answered %d", c.p.Name, res.StatusCode)
			s.record(call)
			appendUsage(r, usage.Record{
				Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
				Provider: c.p.ID, Host: c.p.Where(),
				ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
				ProviderAccount: accountOf(c.p),
				Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
			})
			copyGeminiResponseHeaders(w.Header(), res)
			w.WriteHeader(res.StatusCode)
			w.Write(errBody)
			return
		}

		copyGeminiResponseHeaders(w.Header(), res)
		w.WriteHeader(res.StatusCode)
		snippet := streamAndCapturePrefix(w, res.Body, geminiFileSniffCap)
		res.Body.Close()

		if method == http.MethodDelete && fileName != "" {
			s.geminiState.deleteFile(fileName)
		} else if name := sniffGeminiFileName(snippet); name != "" {
			s.geminiState.saveFile(name, c.p.ID, keyMark(c.p))
		}

		call.Provider = c.p.ID
		call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
		s.record(call)
		appendUsage(r, usage.Record{
			Operation: "files", Time: start, Agent: call.Agent, Via: call.Via,
			Provider: c.p.ID, Host: c.p.Where(),
			ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
			ProviderAccount: accountOf(c.p),
			Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
		})
		return
	}
}

func (s *Server) geminiFileTargets(r *http.Request, fileName string) ([]candidate, int, string) {
	if fileName != "" {
		if owner, ok := s.geminiState.lookupFile(fileName); ok {
			if found, err := provider.Find(owner.providerID); err == nil {
				if q, ok := geminiMarkedKey(*found, owner.keyMark); ok {
					c := candidateOf(q, "")
					if accWho, held := accountHolds(r); held && !accountAllowed(accWho, c) {
						return nil, http.StatusForbidden, keyAccountsError(accWho, fileName)
					}
					if keyWho, held := keyHolds(r); held && !geminiProviderAllowedByKey(keyWho, q) {
						return nil, http.StatusForbidden, keyModelError(keyWho, fileName)
					}
					return []candidate{c}, 0, ""
				}
			}
		}
	}
	cs, keyRefused, errMsg := s.eligibleGeminiCandidates(r, r.Header.Get("X-Magpie-Provider"))
	if keyRefused {
		return nil, http.StatusForbidden, errMsg
	}
	if len(cs) == 0 {
		return nil, http.StatusNotFound, "no enabled Gemini Developer API provider is configured for /v1beta/files"
	}
	return cs, 0, ""
}
