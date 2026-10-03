/* Reviewed manual intents retain their token through refresh and response loss. */
(() => {
  'use strict';
  const form = document.getElementById('manual-order');
  if (!form) return;
  const panel = document.getElementById('order-review');
  const details = document.getElementById('order-details');
  const message = document.getElementById('order-message');
  const confirm = document.getElementById('confirm-order');
  const fresh = document.getElementById('new-order');
  const lookup = document.getElementById('check-order');
  const key = 'manual-order:' + document.body.dataset.namespace;
  let saved = null, busy = false;
  const names = ['kind','symbol','side','qty','sizing','namespace','mode','venue','idempotency_key'];
  function persist() {
    try { sessionStorage.setItem(key, JSON.stringify(saved)); return true; } catch (_) { return false; }
  }
  function draft() { return Object.fromEntries(names.map(name => [name, form.elements.namedItem(name).value])); }
  function restore(values) {
    for (const name of names) {
      const input = form.elements.namedItem(name);
      if (input.tagName === 'SELECT' && !Array.from(input.options).some(option => option.value === values[name])) {
        input.add(new Option(values[name],values[name]));
      }
      input.value = values[name];
    }
  }
  function say(text, error = false, focus = false) {
    message.textContent = text;
    message.setAttribute('role',error ? 'alert' : 'status');
    if (focus) message.focus();
  }
  function display(values) {
    details.replaceChildren();
    for (const [label,value] of values) {
      const term = document.createElement('dt'), description = document.createElement('dd');
      term.textContent=label; description.textContent=value === null ? 'Unavailable' : String(value);
      details.append(term,description);
    }
    panel.hidden=false;
  }
  function showReview() {
    const p=saved.preview;
    display([['Venue / mode',p.venue+' / '+p.mode],['Order',p.kind+' '+p.side+' '+p.symbol],
      ['Server quantity',p.quantity],['Available funds',p.available_funds+' '+p.funding_asset],
      ['Estimated current exposure',p.estimated_exposure],['Estimated execution price',p.estimated_price],
      ['Estimated fee',p.estimated_fee+' '+p.fee_asset],['Settings version',p.config_version],
      ['Strategy',p.strategy_profile+' / '+p.strategy_version],['Price observed',p.price_observed_at],
      ['Protection',p.protection.map(group=>group.state+' '+group.quantity).join('; ') || 'None currently recorded']]);
    confirm.hidden=false; lookup.hidden=!saved.attempted; fresh.hidden=Boolean(saved.attempted);
  }
  function showOperation(operation) {
    saved.can_close=['filled','rejected','canceled','expired'].includes(operation.state);
    persist();
    display([['Operation',operation.operation_id],['Venue / mode',operation.venue+' / '+operation.mode],
      ['Order',operation.side+' '+operation.symbol],['State',operation.state_label],['Reported reason',operation.error_message],
      ['Recovery',operation.next_action],['Confirmed fills',operation.fills.map(fill=>
        fill.qty+' at '+fill.price+'; fee '+(fill.fee_amount ?? 'Unavailable')+' '+(fill.fee_asset || 'asset unavailable')).join('; ') || 'No confirmed fills'],
      ['Protection',operation.protection.map(group=>group.state_label+' '+group.quantity+
        (group.next_action ? ': '+group.next_action : '')).join('; ') || 'None recorded']]);
    confirm.hidden=true; lookup.hidden=false;
    fresh.hidden=!operation.can_start_another;
    say(operation.next_action);
  }
  async function call(url, data) {
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),10000);
    try {
      const response=await fetch(url,{method:data ? 'POST' : 'GET',credentials:'same-origin',cache:'no-store',
        headers:{Accept:'application/json'},body:data,signal:controller.signal});
      const body=await response.json();
      if (!response.ok) {
        const error=new Error((response.status===401 || response.status===403 ?
          'Permission interrupted. Sign in again; this draft and order token are retained. ' : '')+(body.error || 'Request unavailable'));
        error.status=response.status;
        throw error;
      }
      return body;
    } finally { clearTimeout(timeout); }
  }
  async function act(task) {
    if (busy) return;
    busy=true; panel.setAttribute('aria-busy','true');
    try { await task(); } catch (error) { say(error.message+' Check this intent before starting another order.',true,true); }
    finally { busy=false; panel.setAttribute('aria-busy','false'); }
  }
  function freezeDraft(frozen) {
    for (const name of ['symbol','side','qty','sizing']) form.elements.namedItem(name).disabled=frozen;
    form.querySelector('button[type=submit]').disabled=frozen;
    document.getElementById('riskSizeBtn').disabled=frozen;
  }
  form.addEventListener('input',()=> {
    if (saved?.submitted) return;
    saved={draft:draft(),attempted:saved?.attempted}; persist(); confirm.hidden=true;
    say('Draft changed. Review the current inputs before confirming.');
  });
  form.addEventListener('change',()=>form.dispatchEvent(new Event('input')));
  form.addEventListener('submit',event=> {
    event.preventDefault();
    act(async()=> {
      const requestedDraft=draft();
      saved={draft:requestedDraft,attempted:saved?.attempted}; persist();
      const data=new FormData(form);
      const result=await call('/orders/review',data);
      if (JSON.stringify(draft())!==JSON.stringify(requestedDraft)) {
        confirm.hidden=true;
        say('Inputs changed while review was loading. Review the current draft again.');
        return;
      }
      saved={...saved,...result}; persist(); showReview();
      say('Review these server-calculated values. Estimates are not confirmed fills.',false,true);
    });
  });
  confirm.addEventListener('click',()=>act(async()=> {
    if (!saved?.review_token) return;
    saved.submitted=true; saved.attempted=true;
    if (!persist()) {
      saved.submitted=false;
      throw new Error('Browser draft storage is unavailable. Enable storage before confirmation so refresh can retain this intent');
    }
    freezeDraft(true);
    const data=new FormData();
    data.set('csrf_token',form.elements.namedItem('csrf_token').value);
    data.set('review_token',saved.review_token);
    for (const [name,value] of Object.entries(draft())) data.set(name,value);
    let result;
    try { result=await call('/orders/confirm',data); }
    catch (error) {
      // A rejected review can be refreshed with the SAME intake identity.
      // A response loss or permission interruption retains the signed retry.
      if (error.status===409 || error.status===422) {
        saved.submitted=false; saved.review_token=null; persist();
        freezeDraft(false); confirm.hidden=true; fresh.hidden=true;
      }
      throw error;
    }
    saved.operation_id=result.operation_id; persist(); showOperation(result.operation);
  }));
  lookup.addEventListener('click',()=>act(async()=> {
    if (saved?.operation_id) showOperation((await call('/api/operations/'+encodeURIComponent(saved.operation_id))).operation);
    else {
      const result=await call('/api/order-token/'+encodeURIComponent(saved.draft.idempotency_key));
      if (result.operation) {
        saved.operation_id=result.operation.operation_id; persist(); showOperation(result.operation);
      } else say('No saved intent currently found. Review or retry using this retained token; do not start a replacement.',false,true);
    }
  }));
  fresh.addEventListener('click',()=> {
    if (saved?.submitted && fresh.hidden) return;
    saved=null; try { sessionStorage.removeItem(key); } catch (_) {}
    freezeDraft(false); form.elements.namedItem('idempotency_key').value=crypto.randomUUID();
    form.elements.namedItem('kind').value='trade';
    form.elements.namedItem('side').value='buy';
    form.elements.namedItem('sizing').value='quantity';
    panel.hidden=true; say('New draft. Review before submitting.');
  });
  document.getElementById('riskSizeBtn').addEventListener('click',()=>form.dispatchEvent(new Event('input')));
  document.addEventListener('click',event=> {
    const operationButton=event.target.closest('[data-operation-id]');
    if (operationButton) {
      act(async()=> {
        const operation=(await call('/api/operations/'+encodeURIComponent(operationButton.dataset.operationId))).operation;
        const history=document.getElementById('operation-history-detail');
        history.textContent=operation.symbol+' '+operation.state_label+'. '+operation.next_action+' Confirmed fills: '+
          (operation.fills.map(fill=>fill.qty+' at '+fill.price+'; fee '+(fill.fee_amount ?? 'Unavailable')+' '+(fill.fee_asset || 'asset unavailable')).join('; ') || 'None')+
          '. Protection: '+(operation.protection.map(group=>group.state_label+' '+group.quantity).join('; ') || 'None recorded');
        history.focus();
      });
      return;
    }
    const button=event.target.closest('[data-close-symbol]');
    if (!button) return;
    if (saved?.submitted && !saved.can_close) {
      say('Check this pending intent before a separate close. Its unresolved outcome must be reconciled first.',true,true);
      return;
    }
    if (saved?.submitted) {
      saved=null; persist(); freezeDraft(false);
      form.elements.namedItem('idempotency_key').value=crypto.randomUUID();
    }
    form.elements.namedItem('kind').value='close';
    restore({...draft(),kind:'close',symbol:button.dataset.closeSymbol,side:'sell',
      qty:button.dataset.closeQuantity,sizing:'quantity'});
    form.dispatchEvent(new Event('input')); form.elements.namedItem('qty').focus();
    say('Close draft loaded. Review this quantity, or choose the full confirmed holding.');
  });
  try { saved=JSON.parse(sessionStorage.getItem(key) || 'null'); } catch (_) {}
  if (saved?.draft) {
    restore(saved.draft);
    if (saved.preview) showReview();
    if (saved.submitted) {
      freezeDraft(true); lookup.hidden=false; fresh.hidden=true;
      say('Retained submitted intent. Check its status or retry the same confirmation; do not submit a replacement.');
      if (saved.operation_id) lookup.click();
    }
  }
})();
