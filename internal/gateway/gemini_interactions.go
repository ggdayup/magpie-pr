package gateway

import (
	"bufio"
	"bytes"
	"cmp"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/usage"
)

const (
	geminiOwnerCap     = 4096
	geminiOwnerTTL     = 24 * time.Hour
	geminiUploadCap    = 512
	geminiUploadTTL    = 2 * time.Hour
	geminiSniffCap     = 2 << 20
	geminiErrBodyCap   = 1 << 20
	geminiFileSniffCap = 128 << 10
)

type geminiOwner struct {
	providerID string
	keyMark    string
	model      string
	at         time.Time
}

type geminiUploadSession struct {
	upstreamURL string
	providerID  string
	keyMark     string
	at          time.Time
}

// geminiState tracks provider+key affinity for Interactions and Files, and
// opaque tokens for rewritten resumable upload URLs.
type geminiState struct {
	mu           sync.Mutex
	interactions map[string]geminiOwner
	files        map[string]geminiOwner
	uploads      map[string]geminiUploadSession
}

func (g *geminiState) saveInteraction(id, providerID, mark, model string) {
	if id == "" || providerID == "" {
		return
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.interactions == nil {
		g.interactions = map[string]geminiOwner{}
	}
	pruneOwners(g.interactions, now, geminiOwnerTTL, geminiOwnerCap)
	g.interactions[id] = geminiOwner{providerID: providerID, keyMark: mark, model: model, at: now}
}

func (g *geminiState) lookupInteraction(id string) (geminiOwner, bool) {
	if id == "" {
		return geminiOwner{}, false
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	o, ok := g.interactions[id]
	if !ok {
		return geminiOwner{}, false
	}
	if now.Sub(o.at) > geminiOwnerTTL {
		delete(g.interactions, id)
		return geminiOwner{}, false
	}
	return o, true
}

func (g *geminiState) deleteInteraction(id string) {
	if id == "" {
		return
	}
	g.mu.Lock()
	delete(g.interactions, id)
	g.mu.Unlock()
}

func normalizeGeminiFileName(name string) string {
	name = strings.TrimPrefix(strings.TrimSpace(name), "/")
	name = strings.TrimPrefix(name, "files/")
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	return name
}

func (g *geminiState) saveFile(name, providerID, mark string) {
	key := normalizeGeminiFileName(name)
	if key == "" || providerID == "" {
		return
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.files == nil {
		g.files = map[string]geminiOwner{}
	}
	pruneOwners(g.files, now, geminiOwnerTTL, geminiOwnerCap)
	g.files[key] = geminiOwner{providerID: providerID, keyMark: mark, at: now}
}

func (g *geminiState) lookupFile(name string) (geminiOwner, bool) {
	key := normalizeGeminiFileName(name)
	if key == "" {
		return geminiOwner{}, false
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	o, ok := g.files[key]
	if !ok {
		return geminiOwner{}, false
	}
	if now.Sub(o.at) > geminiOwnerTTL {
		delete(g.files, key)
		return geminiOwner{}, false
	}
	return o, true
}

func (g *geminiState) deleteFile(name string) {
	key := normalizeGeminiFileName(name)
	if key == "" {
		return
	}
	g.mu.Lock()
	delete(g.files, key)
	g.mu.Unlock()
}

func (g *geminiState) saveUpload(upstreamURL, providerID, mark string) string {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return ""
	}
	token := hex.EncodeToString(raw[:])
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.uploads == nil {
		g.uploads = map[string]geminiUploadSession{}
	}
	pruneUploads(g.uploads, now, geminiUploadTTL, geminiUploadCap)
	g.uploads[token] = geminiUploadSession{
		upstreamURL: upstreamURL,
		providerID:  providerID,
		keyMark:     mark,
		at:          now,
	}
	return token
}

func (g *geminiState) lookupUpload(token string) (geminiUploadSession, bool) {
	if token == "" {
		return geminiUploadSession{}, false
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	sess, ok := g.uploads[token]
	if !ok {
		return geminiUploadSession{}, false
	}
	if now.Sub(sess.at) > geminiUploadTTL {
		delete(g.uploads, token)
		return geminiUploadSession{}, false
	}
	return sess, true
}

func (g *geminiState) deleteUpload(token string) {
	if token == "" {
		return
	}
	g.mu.Lock()
	delete(g.uploads, token)
	g.mu.Unlock()
}

func pruneOwners(m map[string]geminiOwner, now time.Time, ttl time.Duration, capSize int) {
	if len(m) < capSize {
		return
	}
	for k, v := range m {
		if now.Sub(v.at) > ttl {
			delete(m, k)
		}
	}
	for len(m) >= capSize {
		var oldestKey string
		var oldestAt time.Time
		for k, v := range m {
			if oldestKey == "" || v.at.Before(oldestAt) {
				oldestKey, oldestAt = k, v.at
			}
		}
		if oldestKey == "" {
			break
		}
		delete(m, oldestKey)
	}
}

func pruneUploads(m map[string]geminiUploadSession, now time.Time, ttl time.Duration, capSize int) {
	if len(m) < capSize {
		return
	}
	for k, v := range m {
		if now.Sub(v.at) > ttl {
			delete(m, k)
		}
	}
	for len(m) >= capSize {
		var oldestKey string
		var oldestAt time.Time
		for k, v := range m {
			if oldestKey == "" || v.at.Before(oldestAt) {
				oldestKey, oldestAt = k, v.at
			}
		}
		if oldestKey == "" {
			break
		}
		delete(m, oldestKey)
	}
}

// geminiInteractionsBase returns the upstream /v1beta base URL for a provider
// that speaks the Gemini Developer API (Interactions and Files). Plugins,
// OAuth/subscription accounts, and Vertex AI are excluded.
func geminiInteractionsBase(p provider.Provider) string {
	if !p.On() || p.IsPlugin() || p.Account != nil || p.IsVertex() {
		return ""
	}
	if p.IsRemoteMagpie() {
		base := strings.TrimRight(p.Base(provider.Chat), "/")
		if base == "" {
			return ""
		}
		return strings.TrimSuffix(base, "/v1") + "/v1beta"
	}
	if g := strings.TrimRight(p.Base(provider.Gemini), "/"); g != "" {
		return g
	}
	if isGoogle(p) {
		base := strings.TrimRight(p.Base(provider.Chat), "/")
		return strings.TrimSuffix(base, "/openai")
	}
	return ""
}

// geminiUploadRoot returns the base prefix under which /upload/v1beta/files
// is mounted for p.
func geminiUploadRoot(p provider.Provider) string {
	base := geminiInteractionsBase(p)
	if base == "" {
		return ""
	}
	return strings.TrimSuffix(base, "/v1beta")
}

// geminiMarkedKey returns p configured with the enabled key whose keyMark is
// mark, if p supports the Gemini Developer API.
func geminiMarkedKey(p provider.Provider, mark string) (provider.Provider, bool) {
	if geminiInteractionsBase(p) == "" {
		return p, false
	}
	if p.Key == "" {
		return p, mark == "0"
	}
	for _, k := range p.KeysOn() {
		if q := p.WithKey(k); keyMark(q) == mark && geminiInteractionsBase(q) != "" {
			return q, true
		}
	}
	return p, false
}

// validGeminiID rejects path traversal, slashes, schemes, and control bytes in
// an interaction ID path segment.
func validGeminiID(id string) bool {
	if id == "" || strings.Contains(id, "..") || strings.ContainsAny(id, "/\\\x00\r\n") || strings.Contains(id, "://") {
		return false
	}
	if u, err := url.PathUnescape(id); err != nil || strings.Contains(u, "..") || strings.ContainsAny(u, "/\\\x00\r\n") || strings.Contains(u, "://") {
		return false
	}
	return true
}

// eligibleGeminiCandidates returns enabled Gemini Developer API provider+key
// candidates allowed for r. If provHeader is non-empty, only that provider is
// considered.
func (s *Server) eligibleGeminiCandidates(r *http.Request, provHeader string) (cands []candidate, keyRefused bool, errMsg string) {
	keyWho, keyHeld := keyHolds(r)
	accWho, accHeld := accountHolds(r)

	var provs []provider.Provider
	if provHeader != "" {
		if p, err := provider.Find(strings.TrimSpace(provHeader)); err == nil && geminiInteractionsBase(*p) != "" {
			provs = []provider.Provider{*p}
		}
	} else {
		for _, p := range provider.All() {
			if geminiInteractionsBase(p) != "" {
				provs = append(provs, p)
			}
		}
	}
	if len(provs) == 0 {
		return nil, false, ""
	}

	sawAny := false
	for _, p := range provs {
		if keyHeld && !geminiProviderAllowedByKey(keyWho, p) {
			sawAny = true
			continue
		}
		cs, code, _ := s.onKeys(r, p, "", p.ID)
		if code == http.StatusForbidden {
			sawAny = true
			continue
		}
		for _, c := range cs {
			if geminiInteractionsBase(c.p) == "" {
				continue
			}
			sawAny = true
			if accHeld && !accountAllowed(accWho, c) {
				continue
			}
			cands = append(cands, c)
		}
	}
	if len(cands) == 0 && sawAny && (keyHeld || accHeld) {
		if accHeld {
			return nil, true, keyAccountsError(accWho, "Gemini API")
		}
		return nil, true, keyModelError(keyWho, "Gemini API")
	}
	return probeOne(cands), false, ""
}

// geminiProviderAllowedByKey reports whether a gateway key restricted to some
// models may use p at all (at least one model of p is allowed by who).
func geminiProviderAllowedByKey(who access.Identity, p provider.Provider) bool {
	for _, m := range p.Models {
		if modelAllowed(who, p, m) {
			return true
		}
	}
	for _, m := range p.Available() {
		if modelAllowed(who, p, m.ID) {
			return true
		}
	}
	return modelAllowed(who, p, "")
}

func (s *Server) geminiInteractionsCreate(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	who := callerOf(r)
	call := Call{Time: start, From: provider.Gemini, To: provider.Gemini, Agent: who.agent, Via: who.via, headers: headerLogOf(r.Context())}
	usage.Saw(agentOf(r))
	fail := func(code int, msg string) {
		call.Status, call.Error, call.Millis = code, msg, time.Since(start).Milliseconds()
		writeError(w, provider.Gemini, code, msg)
		s.record(call)
	}

	body, ok := s.requestBody(w, r, provider.Gemini)
	if !ok {
		return
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' || !gjson.ValidBytes(trimmed) {
		fail(http.StatusBadRequest, "request body must be a JSON object")
		return
	}
	modelField := gjson.GetBytes(trimmed, "model").String()
	agentField := gjson.GetBytes(trimmed, "agent").String()
	prevID := gjson.GetBytes(trimmed, "previous_interaction_id").String()
	if modelField == "" && agentField == "" {
		fail(http.StatusBadRequest, "model or agent is required")
		return
	}

	var (
		asked   string
		cands   []candidate
		isGroup bool
		g       provider.Group
		ms      []provider.Member
		grp     *GroupRef
		firstP  provider.Provider
		firstM  string
		isAgent = modelField == ""
	)

	if !isAgent {
		var err error
		asked, err = validateModel(modelField)
		if err != nil {
			fail(http.StatusBadRequest, err.Error())
			return
		}
		call.Model = asked
		noteAsked(r.Context(), asked)
		if msg, off := idOff(asked); off {
			fail(http.StatusForbidden, msg)
			return
		}

		gid := ""
		if provider.IsGroupRef(asked) {
			gid, isGroup = asked, true
		} else if foundGID, ok := provider.GroupFor(asked); ok {
			gid, isGroup = foundGID, true
		}

		if isGroup {
			var found bool
			g, ms, found = provider.FindGroup(gid)
			if !found {
				fail(http.StatusNotFound, unknownModel(asked, ""))
				return
			}
			r = heldByGroup(r, g)
			keyWho, keyHeld := keyHolds(r)
			if !groupGiven(r, keyWho, keyHeld, g, ms) {
				fail(keyModelRefusal(w, keyWho, asked))
				return
			}
			var paused []provider.Paused
			if ms, paused = provider.PausedOut(g.Live(), ms, agentOf(r), "", ruleClock()); len(ms) == 0 && len(paused) > 0 {
				fail(http.StatusServiceUnavailable, pausedError(asked, paused))
				return
			}
			if keyHeld {
				keeps := groupKeeps(keyWho, g, ms)
				var kept []provider.Member
				for _, m := range ms {
					if keeps[m.Provider.ID+"/"+m.Model] {
						kept = append(kept, m)
					}
				}
				if len(kept) == 0 {
					fail(keyGroupRefusal(w, keyWho, asked, ms))
					return
				}
				ms = kept
			}
			grp = groupRef(g, ms)
			cands, _ = s.planGroup(keyRouted(access.Caller(r.Context()), g), ms, provider.Gemini)
			if accWho, accHeld := accountHolds(r); accHeld {
				cands = slices.DeleteFunc(cands, func(c candidate) bool { return !accountAllowed(accWho, c) })
				if len(cands) == 0 {
					fail(http.StatusForbidden, keyAccountsError(accWho, asked))
					return
				}
			}
			cands = slices.DeleteFunc(cands, func(c candidate) bool { return geminiInteractionsBase(c.p) == "" })
			if len(cands) == 0 {
				fail(http.StatusBadRequest, fmt.Sprintf("routing group %q has no Gemini Developer API provider that supports /v1beta/interactions", asked))
				return
			}
			cands = probeOne(cands)
			firstP, firstM = cands[0].p, cands[0].model
		} else {
			p, resolvedModel, ok := provider.Resolve(asked)
			if !ok {
				if off, isOff := provider.SwitchedOff(asked); isOff {
					fail(http.StatusNotFound, switchedOff(off, asked))
					return
				}
				fail(http.StatusNotFound, unknownModel(asked, ""))
				return
			}
			if msg, off := modelOff(p, resolvedModel, asked); off {
				fail(http.StatusForbidden, msg)
				return
			}
			if keyWho, held := keyHolds(r); held && !modelAllowed(keyWho, p, resolvedModel) {
				fail(keyModelRefusal(w, keyWho, asked))
				return
			}
			cs, code, msg := s.onKeys(r, p, resolvedModel, asked)
			if code != 0 {
				fail(code, msg)
				return
			}
			cs = slices.DeleteFunc(cs, func(c candidate) bool { return geminiInteractionsBase(c.p) == "" })
			if len(cs) == 0 {
				fail(http.StatusBadRequest, fmt.Sprintf("%s does not support the Gemini Interactions API (/v1beta/interactions)", p.Name))
				return
			}
			cands = probeOne(cs)
			firstP, firstM = cands[0].p, cands[0].model
		}
	} else {
		asked = agentField
		call.Model = asked
		noteAsked(r.Context(), asked)
		provHint := r.Header.Get("X-Magpie-Provider")
		wireAgent := agentField
		if pid, rest, ok := strings.Cut(agentField, "/"); ok && rest != "" {
			if found, err := provider.Find(pid); err == nil && geminiInteractionsBase(*found) != "" {
				provHint = pid
				wireAgent = rest
			}
		}
		cs, keyRefused, errMsg := s.eligibleGeminiCandidates(r, provHint)
		if keyRefused {
			fail(http.StatusForbidden, errMsg)
			return
		}
		if len(cs) == 0 {
			fail(http.StatusNotFound, "no enabled Gemini Developer API provider is configured for /v1beta/interactions")
			return
		}
		if keyWho, held := keyHolds(r); held {
			cs = slices.DeleteFunc(cs, func(c candidate) bool { return !modelAllowed(keyWho, c.p, wireAgent) })
			if len(cs) == 0 {
				fail(keyModelRefusal(w, keyWho, asked))
				return
			}
		}
		for i := range cs {
			cs[i].model = wireAgent
		}
		cands = cs
		firstP, firstM = cands[0].p, cands[0].model
	}

	// Prefer the provider+key that owns previous_interaction_id when allowed.
	if prevID != "" {
		if owner, ok := s.geminiState.lookupInteraction(prevID); ok {
			cands = s.preferGeminiOwner(r, cands, owner, isAgent)
		}
	}

	// Privacy masking (fails closed) unless every candidate skips redaction.
	allUnmasked := true
	for _, c := range cands {
		if !c.p.SkipsRedaction() {
			allUnmasked = false
			break
		}
	}
	var hits maskHits
	if !allUnmasked {
		rw, maskedBody, done, h, err := redactedHits(maskedFor(r.Context(), firstP, firstM, isGroup, ms), w, body)
		if err != nil {
			fail(maskStatus(err), err.Error())
			return
		}
		defer done()
		w, body, hits = rw, maskedBody, h
	}

	side := s.beginSide(r, start, asked, cands[0].p.ID, grp)
	masked, noted := noteMaskedUntil(r, "interactions", hits, string(body), side.id(), cands[0].p.ID, asked)
	side.masked(masked)
	call.Masked = masked
	call.Route = side.id()
	defer func() { noted(call.Provider, cmp.Or(call.Model, asked)) }()

	var skipped []string
	for i, c := range cands {
		sendBody := body
		if !isAgent {
			wire := c.model
			if !c.p.IsRemoteMagpie() {
				wire = provider.UpstreamNameAsIn(wiresOf(r.Context()), c.p.ID, c.model, c.p.SentIDAt(c.model, ""))
			}
			sendBody = rewriteModel(body, wire)
		} else if c.model != "" && c.model != agentField {
			sendBody = withFields(body, map[string]any{"agent": c.model})
		}

		target := geminiInteractionsBase(c.p) + "/interactions"
		if q := nativeQuery(r.URL.RawQuery); q != "" {
			target += "?" + q
		}

		ctx := s.metered(c.p.Via(r.Context()), c.p, "")
		ctx = provider.WithModel(ctx, c.p.ID, c.model)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(sendBody))
		if err != nil {
			side.end(c.p, c.model, http.StatusBadGateway, err, Usage{})
			fail(http.StatusBadGateway, err.Error())
			return
		}
		copyGeminiRequestHeaders(req.Header, r.Header)
		req.Header.Set("Content-Type", "application/json")
		stampUA(ctx, c.p, req.Header)
		if err := c.p.Sign(ctx, req, provider.Gemini, sendBody); err != nil {
			if i+1 < len(cands) && r.Context().Err() == nil {
				skipped = append(skipped, c.p.ID+": "+err.Error())
				continue
			}
			side.end(c.p, c.model, http.StatusBadGateway, err, Usage{})
			fail(http.StatusBadGateway, err.Error())
			return
		}

		endTry := side.try(c)
		res, err := c.p.Do(s.client, req)
		if err != nil {
			endTry(0, err)
			if i+1 < len(cands) && r.Context().Err() == nil {
				skipped = append(skipped, c.p.ID+": "+err.Error())
				continue
			}
			call.Provider = c.p.ID
			call.Fallback = strings.Join(skipped, "; ")
			call.Millis = time.Since(start).Milliseconds()
			call.Status = http.StatusBadGateway
			call.Error = fmt.Sprintf("%s didn't answer: %v", c.p.Name, err)
			side.end(c.p, c.model, call.Status, errors.New(call.Error), Usage{})
			s.record(call)
			appendUsage(r, usage.Record{
				RouteID: side.id(), Operation: "interactions", Time: start,
				Agent: call.Agent, Via: call.Via, Provider: c.p.ID, Host: c.p.Where(),
				Model: c.model, Requested: asked,
				ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
				ProviderAccount: accountOf(c.p),
				Millis:          call.Millis, Status: call.Status, Session: sessionOf(r.Header),
			})
			writeError(w, provider.Gemini, http.StatusBadGateway, call.Error)
			return
		}

		if res.StatusCode >= 400 {
			errBody, _ := io.ReadAll(io.LimitReader(res.Body, geminiErrBodyCap))
			res.Body.Close()
			upErr := fmt.Errorf("%s answered %d", c.p.Name, res.StatusCode)
			endTry(res.StatusCode, upErr)
			if i+1 < len(cands) && r.Context().Err() == nil && retryable(res.StatusCode, errBody) {
				skipped = append(skipped, fmt.Sprintf("%s: %d", c.p.ID, res.StatusCode))
				continue
			}
			call.Provider = c.p.ID
			call.Fallback = strings.Join(skipped, "; ")
			call.Millis = time.Since(start).Milliseconds()
			call.Status = res.StatusCode
			call.Error = upErr.Error()
			side.end(c.p, c.model, res.StatusCode, upErr, Usage{})
			s.record(call)
			appendUsage(r, usage.Record{
				RouteID: side.id(), Operation: "interactions", Time: start,
				Agent: call.Agent, Via: call.Via, Provider: c.p.ID, Host: c.p.Where(),
				Model: c.model, Requested: asked,
				ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
				ProviderAccount: accountOf(c.p),
				Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
			})
			copyGeminiResponseHeaders(w.Header(), res)
			noteMember(w, r, c.p, c.model)
			w.WriteHeader(res.StatusCode)
			w.Write(errBody)
			return
		}

		endTry(res.StatusCode, nil)
		copyGeminiResponseHeaders(w.Header(), res)
		noteMember(w, r, c.p, c.model)
		w.WriteHeader(res.StatusCode)

		sn := newInteractionSniffer(res.Header.Get("Content-Type"))
		streamAndSniff(w, res.Body, sn)
		res.Body.Close()

		id, served, u := sn.result()
		if id != "" {
			s.geminiState.saveInteraction(id, c.p.ID, keyMark(c.p), c.model)
		}

		call.Provider = c.p.ID
		call.Fallback = strings.Join(skipped, "; ")
		call.Millis = time.Since(start).Milliseconds()
		call.Status = res.StatusCode
		call.Usage = u
		side.end(c.p, c.model, res.StatusCode, nil, u)
		s.record(call)
		appendUsage(r, usage.Record{
			RouteID: side.id(), Operation: "interactions", Time: start,
			Agent: call.Agent, Via: call.Via, Provider: c.p.ID, Host: c.p.Where(),
			Model: c.model, Requested: asked, Served: served,
			ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
			ProviderAccount: accountOf(c.p),
			Input:           u.Input, Output: u.Output, CacheRead: u.CacheRead, Reasoning: u.Reasoning,
			Millis: call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
		})
		return
	}
}

func (s *Server) preferGeminiOwner(r *http.Request, cands []candidate, owner geminiOwner, isAgent bool) []candidate {
	for i, c := range cands {
		if c.p.ID == owner.providerID && keyMark(c.p) == owner.keyMark {
			if i == 0 {
				return cands
			}
			out := make([]candidate, 0, len(cands))
			out = append(out, c)
			out = append(out, cands[:i]...)
			out = append(out, cands[i+1:]...)
			return out
		}
	}
	found, err := provider.Find(owner.providerID)
	if err != nil {
		return cands
	}
	q, ok := geminiMarkedKey(*found, owner.keyMark)
	if !ok {
		return cands
	}
	model := owner.model
	if len(cands) > 0 && cands[0].model != "" {
		model = cands[0].model
	}
	c := candidateOf(q, model)
	if accWho, held := accountHolds(r); held && !accountAllowed(accWho, c) {
		return cands
	}
	if keyWho, held := keyHolds(r); held && !modelAllowed(keyWho, q, model) {
		return cands
	}
	if !isAgent && len(cands) > 0 {
		// Only prepend an external owner if it belongs to the same provider already selected.
		sameProv := false
		for _, existing := range cands {
			if existing.p.ID == q.ID {
				sameProv = true
				break
			}
		}
		if !sameProv {
			return cands
		}
	}
	return append([]candidate{c}, cands...)
}

func (s *Server) geminiInteractionsGet(w http.ResponseWriter, r *http.Request) {
	s.geminiInteractionFollowup(w, r, http.MethodGet, "")
}

func (s *Server) geminiInteractionsCancel(w http.ResponseWriter, r *http.Request) {
	s.geminiInteractionFollowup(w, r, http.MethodPost, "/cancel")
}

func (s *Server) geminiInteractionsDelete(w http.ResponseWriter, r *http.Request) {
	s.geminiInteractionFollowup(w, r, http.MethodDelete, "")
}

func (s *Server) geminiInteractionFollowup(w http.ResponseWriter, r *http.Request, method, suffix string) {
	start := time.Now()
	who := callerOf(r)
	call := Call{Time: start, From: provider.Gemini, To: provider.Gemini, Agent: who.agent, Via: who.via, headers: headerLogOf(r.Context())}
	usage.Saw(agentOf(r))
	fail := func(code int, msg string) {
		call.Status, call.Error, call.Millis = code, msg, time.Since(start).Milliseconds()
		writeError(w, provider.Gemini, code, msg)
		s.record(call)
	}

	id := r.PathValue("id")
	if !validGeminiID(id) {
		fail(http.StatusBadRequest, "invalid interaction id")
		return
	}

	var body []byte
	if method == http.MethodPost {
		var ok bool
		body, ok = s.requestBody(w, r, provider.Gemini)
		if !ok {
			return
		}
	}

	cands, code, errMsg := s.geminiInteractionTargets(r, id)
	if code != 0 {
		fail(code, errMsg)
		return
	}

	for i, c := range cands {
		target := geminiInteractionsBase(c.p) + "/interactions/" + url.PathEscape(id) + suffix
		if q := nativeQuery(r.URL.RawQuery); q != "" {
			target += "?" + q
		}
		ctx := s.metered(c.p.Via(r.Context()), c.p, "")
		ctx = provider.WithModel(ctx, c.p.ID, c.model)
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
			call.Provider, call.Model = c.p.ID, c.model
			call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
			call.Error = fmt.Sprintf("%s answered %d", c.p.Name, res.StatusCode)
			s.record(call)
			appendUsage(r, usage.Record{
				Operation: "interactions", Time: start, Agent: call.Agent, Via: call.Via,
				Provider: c.p.ID, Host: c.p.Where(), Model: c.model, Requested: c.model,
				ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
				ProviderAccount: accountOf(c.p),
				Millis:          call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
			})
			copyGeminiResponseHeaders(w.Header(), res)
			noteMember(w, r, c.p, c.model)
			w.WriteHeader(res.StatusCode)
			w.Write(errBody)
			return
		}

		copyGeminiResponseHeaders(w.Header(), res)
		noteMember(w, r, c.p, c.model)
		w.WriteHeader(res.StatusCode)

		sn := newInteractionSniffer(res.Header.Get("Content-Type"))
		streamAndSniff(w, res.Body, sn)
		res.Body.Close()

		sniffID, served, u := sn.result()
		if method == http.MethodDelete {
			s.geminiState.deleteInteraction(id)
		} else if sniffID != "" {
			s.geminiState.saveInteraction(sniffID, c.p.ID, keyMark(c.p), c.model)
		}
		call.Provider, call.Model = c.p.ID, c.model
		call.Status, call.Millis = res.StatusCode, time.Since(start).Milliseconds()
		call.Usage = u
		s.record(call)
		appendUsage(r, usage.Record{
			Operation: "interactions", Time: start, Agent: call.Agent, Via: call.Via,
			Provider: c.p.ID, Host: c.p.Where(), Model: c.model, Requested: c.model, Served: served,
			ProviderKeyID: provider.KeyID(c.p.Key), ProviderKeyName: c.p.KeyName,
			ProviderAccount: accountOf(c.p),
			Input:           u.Input, Output: u.Output, CacheRead: u.CacheRead, Reasoning: u.Reasoning,
			Millis: call.Millis, Status: res.StatusCode, Session: sessionOf(r.Header),
		})
		return
	}
}

func (s *Server) geminiInteractionTargets(r *http.Request, id string) ([]candidate, int, string) {
	if owner, ok := s.geminiState.lookupInteraction(id); ok {
		if found, err := provider.Find(owner.providerID); err == nil {
			if q, ok := geminiMarkedKey(*found, owner.keyMark); ok {
				c := candidateOf(q, owner.model)
				if accWho, held := accountHolds(r); held && !accountAllowed(accWho, c) {
					return nil, http.StatusForbidden, keyAccountsError(accWho, cmp.Or(owner.model, id))
				}
				if keyWho, held := keyHolds(r); held {
					if owner.model != "" {
						if !modelAllowed(keyWho, q, owner.model) {
							return nil, http.StatusForbidden, keyModelError(keyWho, owner.model)
						}
					} else if !geminiProviderAllowedByKey(keyWho, q) {
						return nil, http.StatusForbidden, keyModelError(keyWho, id)
					}
				}
				return []candidate{c}, 0, ""
			}
		}
	}
	cs, keyRefused, errMsg := s.eligibleGeminiCandidates(r, r.Header.Get("X-Magpie-Provider"))
	if keyRefused {
		return nil, http.StatusForbidden, errMsg
	}
	if len(cs) == 0 {
		return nil, http.StatusNotFound, "no enabled Gemini Developer API provider is configured for /v1beta/interactions"
	}
	return cs, 0, ""
}

func copyGeminiRequestHeaders(dst, src http.Header) {
	for k, vs := range src {
		if !nativeHeaderDropped(k) {
			dst[k] = append([]string(nil), vs...)
		}
	}
}

func copyGeminiResponseHeaders(dst http.Header, res *http.Response) {
	for k, vs := range res.Header {
		ck := http.CanonicalHeaderKey(k)
		if nativeAnswerDropped[ck] || (res.Uncompressed && (ck == "Content-Length" || ck == "Content-Encoding")) {
			continue
		}
		dst[k] = append([]string(nil), vs...)
	}
}

func streamAndSniff(w http.ResponseWriter, src io.Reader, sn *interactionSniffer) {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32<<10)
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if sn != nil {
				sn.write(buf[:n])
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			rc.Flush()
		}
		if err != nil {
			return
		}
	}
}

// interactionSniffer extracts interaction id, served model, and token usage
// from either a JSON response or an SSE stream without buffering large streams.
type interactionSniffer struct {
	sse     bool
	lineBuf bytes.Buffer
	jsonBuf bytes.Buffer
	id      string
	model   string
	usage   Usage
}

func newInteractionSniffer(contentType string) *interactionSniffer {
	return &interactionSniffer{
		sse: strings.HasPrefix(strings.ToLower(strings.TrimSpace(contentType)), "text/event-stream"),
	}
}

func (s *interactionSniffer) write(p []byte) {
	if s.sse {
		s.lineBuf.Write(p)
		for {
			b := s.lineBuf.Bytes()
			i := bytes.IndexByte(b, '\n')
			if i < 0 {
				if s.lineBuf.Len() > geminiSniffCap {
					s.lineBuf.Reset()
				}
				return
			}
			line := string(bytes.TrimRight(b[:i], "\r"))
			s.lineBuf.Next(i + 1)
			if data, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(data)
				if data != "" && data != "[DONE]" && data[0] == '{' {
					s.parseObject([]byte(data))
				}
			}
		}
	}
	if s.jsonBuf.Len() < geminiSniffCap {
		rem := geminiSniffCap - s.jsonBuf.Len()
		if len(p) > rem {
			p = p[:rem]
		}
		s.jsonBuf.Write(p)
	}
}

func (s *interactionSniffer) result() (id, model string, u Usage) {
	if s.sse {
		if s.lineBuf.Len() > 0 {
			sc := bufio.NewScanner(&s.lineBuf)
			for sc.Scan() {
				if data, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "data:"); ok {
					data = strings.TrimSpace(data)
					if data != "" && data != "[DONE]" && data[0] == '{' {
						s.parseObject([]byte(data))
					}
				}
			}
		}
	} else if s.jsonBuf.Len() > 0 {
		s.parseObject(s.jsonBuf.Bytes())
	}
	return s.id, s.model, s.usage
}

func (s *interactionSniffer) parseObject(raw []byte) {
	if !gjson.ValidBytes(raw) {
		return
	}
	root := gjson.ParseBytes(raw)
	if id := cmp.Or(root.Get("id").String(), root.Get("interaction.id").String()); id != "" {
		s.id = id
	}
	if m := cmp.Or(root.Get("model").String(), root.Get("interaction.model").String()); m != "" {
		s.model = m
	}
	for _, path := range []string{"usage", "total_usage", "interaction.usage", "interaction.total_usage"} {
		u := root.Get(path)
		if !u.IsObject() {
			continue
		}
		in := firstInt(u, "total_input_tokens", "input_tokens")
		cached := firstInt(u, "total_cached_tokens", "cached_tokens")
		out := firstInt(u, "total_output_tokens", "output_tokens")
		thought := firstInt(u, "total_thought_tokens", "thought_tokens")
		if in > 0 || cached > 0 {
			s.usage.Input = max(in-cached, 0)
			s.usage.CacheRead = cached
		}
		if out > 0 || thought > 0 {
			s.usage.Output = out + thought
			s.usage.Reasoning = thought
		}
	}
}

func firstInt(obj gjson.Result, keys ...string) int {
	for _, k := range keys {
		if r := obj.Get(k); r.Exists() {
			return int(r.Int())
		}
	}
	return 0
}
