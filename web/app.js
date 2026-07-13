// HSK-5 tutor — page JS, run once at load. Loaded by app.py and injected
// via launch(head=) as a self-invoking <script>. __STARTERS__ is replaced
// by app.py with the model-written starter pool (JSON) at startup.

(() => {
  const root = document.documentElement;
  root.classList.remove('dark');
  new MutationObserver(() => root.classList.remove('dark'))
    .observe(root, { attributes: true, attributeFilter: ['class'] });

  const KEY = 'hsk5-tutor-deck-v1';   // shared with web/flashcards.html
  const loadDeck = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
  // Writing to a Gradio-bound textarea needs the native setter + an input
  // event, or Svelte's store never sees the change — one helper, used by every
  // programmatic write (ask box, deck mirror, card requests, starter chips).
  const setNative = (el, value) => {
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')
      .set.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const toast = (msg) => {
    let t = document.getElementById('collect-toast');
    if (!t) { t = document.createElement('div'); t.id = 'collect-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2000);
  };
  const plainText = (el) => {   // textContent with the ruby pinyin stripped
    const c = el.cloneNode(true);
    c.querySelectorAll('rt').forEach(r => r.remove());
    return c.textContent;
  };

  document.addEventListener('click', (e) => {
    const hz = e.target.closest('.hz');
    if (!hz) return;
    const front = plainText(hz).trim();
    if (!front) return;
    const deck = loadDeck();
    if (deck.some(c => c.front === front)) { toast('“' + front + '” 已在卡片里 · already in your deck'); return; }
    const tip = hz.dataset.tip || '';
    const [pinyin, gloss = ''] = tip.split(/ — (.*)/s, 2);
    // placeholder example: the sentence around the word, from the same bubble —
    // must be mostly Chinese (a quoted word inside an English sentence used to
    // slip through). Replaced by a model-written example via #card-req below.
    const msg = hz.closest('.msg, .rd-passage');
    const sentence = msg
      ? (plainText(msg).match(/[^。！？!?\n]*[。！？!?]?/g) || []).find(s =>
          s.includes(front) && (s.match(/[一-鿿]/g) || []).length / s.trim().length > 0.4)
      : '';
    const card = { id: front + ':' + Date.now(), front, pinyin, gloss,
                   example: (sentence || '').trim().slice(0, 120),
                   ease: 2.5, interval: 0, reps: 0, lapses: 0, due: 0 };
    deck.push(card);
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已收藏 “' + front + '” · added to your deck (' + deck.length + ')');
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
    requestExample(card);
  });

  // Ask the server to write a fresh example sentence for a new card; the reply
  // shows up in #card-res (watched by the observer below).
  const requestExample = (card) => {
    const ta = document.querySelector('#card-req textarea');
    if (ta) setNative(ta, JSON.stringify({ id: card.id, word: card.front, gloss: card.gloss || '' }));
  };

  // Mirror every deck change to the server's data/deck.json (debounced —
  // a review session rates cards in quick bursts).
  let pushT;
  const pushDeckFile = () => {
    clearTimeout(pushT);
    pushT = setTimeout(() => {
      const ta = document.querySelector('#deck-save textarea');
      if (ta) setNative(ta, localStorage.getItem(KEY) || '[]');
    }, 600);
  };
  // Restore: a browser with NO deck key at all (fresh browser / cleared site
  // data) adopts the server file. An empty-but-present deck is respected —
  // deleting your last card doesn't resurrect it on reload.
  const ensureDeckRestore = () => {
    const el = document.getElementById('deck-file');
    if (!el || el.dataset.done) return;
    el.dataset.done = '1';
    if (localStorage.getItem(KEY) !== null) return;
    const text = el.textContent.trim();
    if (!text) return;
    try {
      if (Array.isArray(JSON.parse(text))) {
        localStorage.setItem(KEY, text);
        renderWordlist();
        syncDeckWords();
      }
    } catch {}
  };
  let lastCardRes = '';
  const checkCardRes = () => {
    const el = document.getElementById('card-res');
    const text = el ? el.textContent.trim() : '';
    if (!text || text === lastCardRes) return;
    lastCardRes = text;
    let res;
    try { res = JSON.parse(text); } catch { return; }
    if (!res.id) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === res.id);
    if (!card) return;                     // removed before the model finished
    let got = [];
    if (res.example) {
      card.example = res.example;
      card.example_en = res.example_en || '';
      got.push('例句');
    }
    if (res.gloss && !card.gloss) {        // model-written definition fills an empty slot only
      card.gloss = res.gloss;
      got.push('释义');
    }
    if (!got.length) return;
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast(got.join('和') + '写好了 · card filled in for “' + card.front + '”');
    renderWordlist();
    pushDeckFile();
  };

  // Correction cards: the server tags correctable tutor bubbles with a
  // .fix-collect chip; saving stores a kind:'fix' card (front = the student's
  // wrong sentence, back = the fix + rule) in the same deck/scheduler.
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.fix-collect');
    if (!chip) return;
    const { wrong, fix, why } = chip.dataset;
    if (!wrong || !fix) return;
    const deck = loadDeck();
    if (deck.some(c => c.kind === 'fix' && c.front === wrong && c.fix === fix)) {
      toast('这条纠错已经在卡片里 · correction already saved');
      chip.classList.add('saved');
      return;
    }
    deck.push({ id: 'fix:' + Date.now(), kind: 'fix', front: wrong, fix, why: why || '',
                ease: 2.5, interval: 0, reps: 0, lapses: 0, due: 0 });
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已收藏纠错 · correction saved (' + deck.length + ')');
    chip.classList.add('saved');
    chip.textContent = '✓ 已收藏 · saved';
    renderWordlist();
    pushDeckFile();
  });

  // ---- word-list tab: table of the collected deck, with per-row removal.
  // Re-rendered whenever the deck changes: after a collect (above), after a
  // removal (below), and on `storage` events from the flashcards iframe
  // (ratings, manual adds, resets). Removals propagate back the same way.
  const escHtml = (s) => String(s).replace(/[&<>"]/g,
    (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
  const renderWordlist = () => {
    const el = document.getElementById('wordlist');
    if (!el) return;
    // don't re-render out from under an in-progress edit
    if (document.activeElement && document.activeElement.closest
        && document.activeElement.closest('.wl-edit')) return;
    const deck = loadDeck();
    if (!deck.length) {
      el.innerHTML = '<div class="wl-empty">生词表是空的 — 在对话里点一个词就能收藏。<br>' +
        'Nothing collected yet — click any word in the chat to add it.</div>';
      return;
    }
    // gloss/example/translation (fix/why on correction cards) are editable in place
    const editable = (c, field, extra) =>
      ' class="wl-edit' + (extra ? ' ' + extra : '') + '" contenteditable="true" spellcheck="false"'
      + ' data-fid="' + escHtml(c.id) + '" data-field="' + field + '"'
      + ' title="点击编辑 · click to edit"';
    const edit = (c, field, cls, val) =>
      '<td' + editable(c, field, cls) + '>' + escHtml(val || '') + '</td>';
    const rows = [...deck].reverse().map(c => {
      const open = ' class="wl-open" data-fid="' + escHtml(c.id)
        + '" title="打开卡片 · open this card"';
      const cells = c.kind === 'fix'
        ? '<td class="hanzi sent"><span' + open + '>' + escHtml(c.front) + '</span></td>'
          + '<td class="py">改错</td>'
          + edit(c, 'fix', 'fixto', c.fix)
          + edit(c, 'why', 'ex', c.why)
        : '<td class="hanzi"><span' + open + '>' + escHtml(c.front) + '</span></td>'
          + '<td class="py">' + escHtml(c.pinyin || '') + '</td>'
          + edit(c, 'gloss', '', c.gloss)
          // example + its translation as two separately-editable blocks
          + '<td class="ex"><div' + editable(c, 'example') + '>' + escHtml(c.example || '') + '</div>'
          + '<div' + editable(c, 'example_en', 'wl-en') + ' data-ph="translation…">'
          + escHtml(c.example_en || '') + '</div></td>';
      return '<tr>' + cells
        + '<td class="st">' + (c.reps > 0 ? c.reps + '×' : 'new') + '</td>'
        + '<td><button class="wl-remove" title="移除 · remove" data-fid="'
        + escHtml(c.id) + '">✕</button></td></tr>';
    }).join('');
    el.innerHTML =
      '<div class="wl-head">生词表 · collected words <b>' + deck.length + '</b></div>'
      + '<table class="wl-table"><thead><tr><th>词</th><th>拼音</th><th>释义</th>'
      + '<th>例句</th><th>复习</th><th></th></tr></thead><tbody>' + rows + '</tbody></table>';
  };
  // in-place edits: save on blur; Enter commits instead of inserting a newline
  document.addEventListener('focusout', (e) => {
    const td = e.target.closest('.wl-edit');
    if (!td) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === td.dataset.fid);
    if (!card) return;
    const val = td.textContent.trim();
    if ((card[td.dataset.field] || '') === val) return;
    card[td.dataset.field] = val;
    localStorage.setItem(KEY, JSON.stringify(deck));
    toast('已保存 · saved');
    pushDeckFile();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.closest('.wl-edit')) {
      e.preventDefault();
      e.target.closest('.wl-edit').blur();
    }
  });

  // clicking a word in the list opens its card in the flashcards tab; the
  // message is sent twice because a first-ever visit mounts the iframe lazily
  document.addEventListener('click', (e) => {
    const word = e.target.closest('.wl-open');
    if (!word) return;
    [...document.querySelectorAll('button[role="tab"]')]
      .find(t => t.textContent.includes('卡片'))?.click();
    const show = () => document.querySelector('.cards-frame')
      ?.contentWindow?.postMessage({ type: 'show-card', id: word.dataset.fid }, '*');
    setTimeout(show, 250);
    setTimeout(show, 900);
  });

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.wl-remove');
    if (!btn) return;
    const deck = loadDeck();
    const card = deck.find(c => c.id === btn.dataset.fid);
    localStorage.setItem(KEY, JSON.stringify(deck.filter(c => c.id !== btn.dataset.fid)));
    toast('已移除 “' + (card ? card.front : '') + '” · removed');
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
  });

  // Mirror the deck's word fronts (due-first, capped) into the hidden #deck-words
  // textbox so the server's pick_targets() can prefer the student's own words.
  const syncDeckWords = () => {
    const ta = document.querySelector('#deck-words textarea');
    if (!ta) return;
    const now = Date.now();
    const words = loadDeck().filter(c => c.kind !== 'fix');  // fix fronts are sentences
    setNative(ta, JSON.stringify({
      due: words.filter(c => c.due <= now).map(c => c.front).slice(0, 30),
      other: words.filter(c => c.due > now).map(c => c.front).slice(0, 30),
    }));
  };
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY) return;
    renderWordlist();
    syncDeckWords();
    pushDeckFile();
  });
  // The mirror's real guarantee: re-sync when the ask box gains focus — the
  // user focuses before typing, giving Gradio's (async) store update seconds
  // to settle before submit. (A submit-instant sync was tested and loses the
  // race: the DOM updates but Gradio snapshots its store value first.)
  document.addEventListener('focusin', (e) => {
    if (e.target.closest('#ask textarea')) syncDeckWords();
  });

  // ---- Text-to-speech: neural (edge-tts via the server) with a browser fallback.
  // Each .spk button's data-speak carries the Chinese-only text. A click asks the
  // server (#tts-req) for an MP3; the reply lands in #tts-res and plays. If the
  // server can't (offline / edge-tts missing / slow), we use the browser voice.
  let zhVoice = null;
  const pickVoice = () => {
    const vs = window.speechSynthesis ? speechSynthesis.getVoices() : [];
    zhVoice = vs.find(v => /^zh/i.test(v.lang)) || null;
  };
  if (window.speechSynthesis) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
  let ttsAudio = null, ttsPending = null, ttsNonce = 0, ttsSpeaking = false;
  const ttsRate = () => {
    const s = document.getElementById('tts-speed');
    const v = s ? parseFloat(s.value) : 1;
    return (v >= 0.5 && v <= 2) ? v : 1;
  };
  document.addEventListener('input', (e) => {
    if (e.target.id !== 'tts-speed') return;
    const v = ttsRate();
    const lbl = document.getElementById('speed-val');
    if (lbl) lbl.textContent = v.toFixed(1) + '×';
    if (ttsAudio) ttsAudio.playbackRate = v;      // live-adjust a clip already playing
  });
  const browserSpeak = (text) => {
    if (!window.speechSynthesis || !text) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN'; u.rate = ttsRate(); if (zhVoice) u.voice = zhVoice;
    u.onend = u.onerror = () => { ttsSpeaking = false; };
    ttsSpeaking = true;
    speechSynthesis.speak(u);
  };
  const stopTts = () => {
    if (ttsAudio) { ttsAudio.pause(); ttsAudio = null; }
    if (window.speechSynthesis) speechSynthesis.cancel();
    ttsSpeaking = false;
    if (ttsPending) { ttsPending.btn.classList.remove('spk-loading'); ttsPending = null; }
  };
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.spk');
    if (!btn) return;
    // something OF OURS already playing or pending? this click just stops it.
    // We track our own ttsSpeaking flag instead of speechSynthesis.speaking,
    // which can read spuriously-true (e.g. no audio device) and wedge the guard.
    if (ttsAudio || ttsPending || ttsSpeaking) {
      stopTts();
      return;
    }
    const text = btn.dataset.speak || '';
    if (!text) return;
    const ta = document.querySelector('#tts-req textarea');
    if (!ta) { browserSpeak(text); return; }            // channel not mounted -> browser
    const id = ++ttsNonce;
    ttsPending = { id, btn };
    btn.classList.add('spk-loading');
    setNative(ta, JSON.stringify({ id, text }));
    setTimeout(() => {                                   // server too slow -> browser
      if (ttsPending && ttsPending.id === id) { stopTts(); browserSpeak(text); }
    }, 6000);
  });
  let lastTtsRes = '';
  const checkTtsRes = () => {
    const el = document.getElementById('tts-res');
    const t = el ? el.textContent.trim() : '';
    if (!t || t === lastTtsRes) return;
    lastTtsRes = t;
    let res; try { res = JSON.parse(t); } catch { return; }
    if (!ttsPending || res.id !== ttsPending.id) return;   // stale / superseded
    if (ttsPending.iframe) {   // request came from the flashcards iframe — hand it back
      ttsPending.iframe.postMessage({ type: 'tts-audio', id: res.id,
        audio: res.audio || null, fail: !!(res.fail || !res.audio), rate: ttsRate() }, '*');
      ttsPending = null;
      return;
    }
    const btn = ttsPending.btn; ttsPending = null;
    btn.classList.remove('spk-loading');
    if (res.fail || !res.audio) { browserSpeak(btn.dataset.speak || ''); return; }
    ttsAudio = new Audio(res.audio);
    try { ttsAudio.preservesPitch = true; } catch (_) {}   // speed change keeps pitch
    ttsAudio.playbackRate = ttsRate();
    ttsAudio.onended = ttsAudio.onerror = () => { ttsAudio = null; };
    ttsAudio.play().catch(() => { ttsAudio = null; browserSpeak(btn.dataset.speak || ''); });
  };

  // ---- voice input: Web Speech API (Chrome; the button only appears when the
  // API exists). Click to talk in Mandarin — interim results stream into the
  // ask box (appended to whatever's typed), stop on click or when you pause.
  // Recognition text is never auto-submitted: mis-hearings should be read (and
  // fixed) by the learner before they go to the tutor.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null;
  const setAsk = (text) => {
    const ta = document.querySelector('#ask textarea');
    if (ta) setNative(ta, text);
  };
  const ensureMic = () => {
    if (!SR) return;
    // anchor inside the input bar itself (the submit button's container) so
    // top:50% centers against the bar, not the padded outer block
    const box = document.querySelector('#ask .input-container') || document.querySelector('#ask');
    if (!box || box.querySelector('.mic-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'mic-btn'; btn.type = 'button';
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none"'
      + ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
      + '<rect x="9" y="2" width="6" height="12" rx="3"/>'
      + '<path d="M5 10v1a7 7 0 0 0 14 0v-1"/><path d="M12 18v3"/></svg>';
    btn.title = '点一下，说中文 · click and speak Chinese';
    box.appendChild(btn);
    btn.addEventListener('click', () => {
      if (rec) { rec.stop(); return; }        // click again to stop
      rec = new SR();
      rec.lang = 'zh-CN'; rec.interimResults = true; rec.continuous = false;
      const ta = document.querySelector('#ask textarea');
      const base = ta ? ta.value : '';
      rec.onresult = (e) => {
        let heard = '';
        for (const r of e.results) heard += r[0].transcript;
        setAsk(base + heard);
      };
      rec.onerror = (e) => {
        const why = { 'not-allowed': '需要麦克风权限 · mic permission needed',
                      'no-speech': '没听到声音 · heard nothing' }[e.error] || e.error;
        toast('语音输入 · voice input: ' + why);
      };
      rec.onend = () => {
        rec = null; btn.classList.remove('rec');
        const t = document.querySelector('#ask textarea');
        if (t) t.focus();
      };
      btn.classList.add('rec');
      rec.start();
    });
  };

  // Starter chips: a fresh random six from the pool on every page load; the ⟲
  // button asks the server to write a genuinely new set (fresh random seeds).
  const STARTER_POOL = __STARTERS__;
  const renderStarters = (row, picks) => {
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    row.innerHTML = '<span class="starters-label">试一试 · try one</span>'
      + picks.map(p => '<button class="starter-chip">' + esc(p) + '</button>').join('')
      + '<button class="starter-refresh" title="换一批 · write me a new six">⟲</button>';
  };
  const ensureStarters = () => {
    const row = document.getElementById('starters');
    if (!row || row.dataset.filled) return;
    row.dataset.filled = '1';
    renderStarters(row, [...STARTER_POOL].sort(() => Math.random() - 0.5).slice(0, 6));
  };
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.starter-refresh');
    if (!btn || btn.classList.contains('busy')) return;
    btn.classList.add('busy');
    const ta = document.querySelector('#starters-req textarea');
    if (ta) setNative(ta, String(Date.now()));
  });
  let lastStarters = '';
  const checkStartersRes = () => {
    const el = document.getElementById('starters-data');
    const text = el ? el.textContent.trim() : '';
    if (!text || text === lastStarters) return;
    lastStarters = text;
    let picks;
    try { picks = JSON.parse(text); } catch { return; }
    const row = document.getElementById('starters');
    if (Array.isArray(picks) && picks.length && row) renderStarters(row, picks);
  };
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.starter-chip');
    if (!chip) return;
    const ta = document.querySelector('#ask textarea');
    if (!ta) return;
    setNative(ta, chip.textContent);
    ta.focus();
  });

  // reading tab: reveal / hide a comprehension answer
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.rd-reveal');
    if (!btn) return;
    const a = btn.parentElement.querySelector('.rd-a');
    if (!a) return;
    a.hidden = !a.hidden;
    btn.textContent = a.hidden ? '显示答案 · show answer' : '隐藏答案 · hide answer';
  });

  // 问老师 from a flashcard back (the cards iframe posts {type:'ask-tutor'}):
  // switch to the chat tab, fill the ask box, submit. Only messages from OUR
  // iframe are honored (e.source check — srcdoc frames have origin 'null', so
  // source identity is the usable credential). This path submits without the
  // user ever focusing the ask box, so it must sync the deck mirror itself —
  // first thing, giving the store the full 600ms of the two beats below to
  // settle (the same lead-time lesson as the focusin listener above).
  window.addEventListener('message', (e) => {
    const cards = document.querySelector('.cards-frame');
    if (!cards || e.source !== cards.contentWindow || !e.data) return;
    // Neural-TTS proxy for the flashcards iframe: run its request through our
    // channel and post the audio back (at the current speed + selected voice).
    if (e.data.type === 'tts-request' && typeof e.data.text === 'string') {
      stopTts();
      const ta = document.querySelector('#tts-req textarea');
      if (!ta) { cards.contentWindow.postMessage({ type: 'tts-audio', id: e.data.id, fail: true }, '*'); return; }
      ttsPending = { id: e.data.id, iframe: cards.contentWindow };
      const v = (e.data.voice === 'male' || e.data.voice === 'female') ? e.data.voice : undefined;
      setNative(ta, JSON.stringify({ id: e.data.id, text: e.data.text, voice: v }));
      return;
    }
    if (e.data.type !== 'ask-tutor' || typeof e.data.text !== 'string') return;
    syncDeckWords();
    const chatTab = [...document.querySelectorAll('button[role="tab"]')]
      .find(t => t.textContent.includes('对话'));
    if (chatTab) chatTab.click();
    setTimeout(() => {
      setAsk(e.data.text);
      setTimeout(() => document.querySelector('#ask .submit-button')?.click(), 300);
    }, 300);
  });

  // The transcript is capped (overflow-y): jump to the newest message when one
  // arrives. Three traps, all hit in testing:
  //  - Gradio 6 patches the .chat node IN PLACE on update (it is not replaced),
  //    so new messages are detected by bubble count — which also means user
  //    scroll position survives everything except a new message, as it should;
  //  - this script runs in <head>, where document.body is still null — the
  //    observer must be installed after DOM ready or observe() throws;
  //  - the jump must repeat as layout settles: mutations fire pre-layout, and
  //    the web-font swap can grow scrollHeight again afterwards.
  // The same observer fills the starter row once Gradio mounts it (the Svelte
  // app renders well after DOMContentLoaded).
  const installObserver = () => {
    let lastCount = -1;
    const jumpToNewest = (chat) => {
      const jump = () => { chat.scrollTop = chat.scrollHeight; };
      requestAnimationFrame(jump);
      setTimeout(jump, 120);
      setTimeout(jump, 500);
    };
    // one-shot initial fill once Gradio mounts the container (guarded via
    // data-filled so our own innerHTML writes don't re-trigger through the
    // observer)
    const ensureWordlist = () => {
      const el = document.getElementById('wordlist');
      if (!el || el.dataset.filled) return;
      el.dataset.filled = '1';
      renderWordlist();
    };
    const ensureDeckSync = () => {
      const ta = document.querySelector('#deck-words textarea');
      if (!ta || ta.dataset.synced) return;
      ta.dataset.synced = '1';
      // best-effort early sync; the real guarantees are the focusin listener
      // on the ask box and the ask-tutor message handler (both re-sync with
      // lead time before their submits)
      syncDeckWords();
    };
    let lastHeight = 0;
    new MutationObserver(() => {
      ensureStarters();
      ensureWordlist();
      ensureDeckSync();
      ensureDeckRestore();
      ensureMic();
      checkCardRes();
      checkStartersRes();
      checkTtsRes();
      const chat = document.querySelector('.chat');
      if (!chat) return;
      if (chat.childElementCount !== lastCount) {
        lastCount = chat.childElementCount;
        jumpToNewest(chat);
      } else if (chat.scrollHeight !== lastHeight
                 && chat.scrollHeight - chat.scrollTop - chat.clientHeight < 160) {
        // streaming growth: stay pinned to the bottom unless the user scrolled up
        chat.scrollTop = chat.scrollHeight;
      }
      lastHeight = chat.scrollHeight;
    }).observe(document.body, { childList: true, subtree: true });
    ensureStarters();
    ensureWordlist();
    ensureDeckSync();
    ensureDeckRestore();
    ensureMic();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installObserver);
  } else {
    installObserver();
  }
})();
