/* Real Chromium, guarded loopback HTTP, transactional owner, fake providers. */
const {chromium,expect}=require('@playwright/test');
const {default:AxeBuilder}=require('@axe-core/playwright');
const {spawn,execFileSync}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
(async()=> {
  const evidence=path.resolve('docs/planning/trading-remediation/evidence/bt13-browser');
  fs.mkdirSync(evidence,{recursive:true});
  const browser=await chromium.launch(),results=[];
  try {
    for (const scenario of (process.argv.length>2 ? process.argv.slice(2) : ['paper','partial','unknown','reject','protection_failed','fill'])) {
      const state=fs.mkdtempSync(path.join(os.tmpdir(),'bt13-browser-'));
      const server=spawn('python',['-m','tests.operator_browser_server',state,'0',scenario],{windowsHide:true});
      let output='';server.stdout.on('data',data=>output+=data);server.stderr.on('data',data=>output+=data);
      const context=await browser.newContext({viewport:{width:390,height:844},reducedMotion:'reduce'});
      try {
        const start=Date.now();
        while(!output.match(/ready: (http:\/\/127\.0\.0\.1:\d+)/)) {
          if(server.exitCode!==null || Date.now()-start>30000) throw new Error('Fixture startup failed: exit='+server.exitCode+' '+output);
          await new Promise(resolve=>setTimeout(resolve,100));
        }
        const url=output.match(/ready: (http:\/\/127\.0\.0\.1:\d+)/)[1];
        await context.route('**/*',route=>new URL(route.request().url()).origin===url ? route.continue() : route.abort());
        const page=await context.newPage(),errors=[];
        page.on('pageerror',error=>errors.push(error.message));
        async function login(target=page) {
          await target.goto(url+'/login');await target.locator('[name=username]').fill('operator');
          await target.locator('[name=password]').fill('offline-password');
          await target.locator('button[type=submit]').click();await target.waitForURL(url+'/');
          await expect(target.locator('#connection-status')).not.toHaveText('Status loaded.');
        }
        async function fixture(action) {
          const csrf=await page.locator('[name=csrf_token]').first().inputValue();
          const reply=await context.request.post(url+'/__browser_fixture',{form:{csrf_token:csrf,action}});
          expect(reply.status(),await reply.text()).toBe(200);
        }
        async function status() {return (await context.request.get(url+'/api/status')).json();}
        async function refresh() {
          await page.locator('#refreshStatus').click();
          await expect(page.locator('#connection-status')).not.toHaveAttribute('aria-busy','true');
        }
        async function review() {
          await page.locator('#manual-order button[type=submit]').click();
          try {await expect(page.locator('#confirm-order')).toBeVisible();}
          catch(error) {throw new Error(scenario+' review: '+await page.locator('#order-message').textContent()+'; browser errors '+JSON.stringify(errors)+'; '+error.message);}
          await expect(page.locator('#order-details')).toContainText('Server quantity');
          await expect(page.locator('#order-message')).toBeFocused();
        }
        async function confirm() {
          await page.locator('#confirm-order').click();
          await expect(page.locator('#order-details')).toContainText('Queued for the execution owner');
          await expect(page.locator('#order-details')).toContainText('No confirmed fills');
        }
        await login();
        if(scenario==='paper') {
          const inputs={initial_cash:'10000',fee_fraction:'0',slippage_fraction:'0',risk_fraction:'1',fixed_quote:'100',
            max_allocation_fraction:'1',max_exposure_fraction:'1',max_positions:'100',cooldown_seconds:'0',drawdown_fraction:'1'};
          for(const [name,value] of Object.entries(inputs)) await page.locator('#setup-'+name).fill(value);
          await page.locator('form[action="/paper/setup"] button[type=submit]').click();
          await expect(page.locator('[data-status-panel=setup-status]')).toContainText('Explicit operator policy recorded');
          await fixture('scan');await refresh();
        }
        await page.locator('#qtyInput').fill('1');await review();
        expect((await status()).operations).toEqual([]);
        const originalKey=await page.locator('#manual-order [name=idempotency_key]').inputValue();
        if(scenario==='paper') {
          await fixture('mark');await page.locator('#confirm-order').click();
          await expect(page.locator('#order-message')).toContainText('changed');
          await expect(page.locator('#order-message')).toBeFocused();
          await expect(page.locator('#qtyInput')).toHaveValue('1');
          await expect(page.locator('#manual-order [name=idempotency_key]')).toHaveValue(originalKey);
          await review();
          // Lose the actual response after committed intake; do not fake a fill.
          await page.route('**/orders/confirm',async route=>{await route.fetch();await route.abort();});
          await page.locator('#confirm-order').click();
          await expect(page.locator('#order-message')).toHaveAttribute('role','alert');
          expect((await status()).operations).toHaveLength(1);
          await page.unroute('**/orders/confirm');await page.reload();
          await expect(page.locator('#qtyInput')).toHaveValue('1');
          await expect(page.locator('#manual-order [name=idempotency_key]')).toHaveValue(originalKey);
          await page.locator('#check-order').click();
          await expect(page.locator('#order-details')).toContainText('Queued for the execution owner');
          expect((await status()).operations).toHaveLength(1);
        } else await confirm();
        await fixture('process');await page.locator('#check-order').click();
        const wanted={paper:'Filled',partial:'Partly filled',unknown:'unknown',reject:'Rejected',protection_failed:'Protection failed',fill:'Filled'}[scenario];
        await expect(page.locator('#order-details')).toContainText(wanted);
        if(['partial','unknown','protection_failed'].includes(scenario)) await expect(page.locator('#new-order')).toBeHidden();
        if(scenario==='partial') await expect(page.locator('#order-details')).toContainText('0.4 at 101; fee 0.0808 USDT');
        if(scenario==='unknown') {
          await expect(page.locator('#order-details')).toContainText('No confirmed fills');
          await page.reload();await page.locator('#check-order').click();
          await expect(page.locator('#order-details')).toContainText('unknown');
          await fixture('reconcile');await page.locator('#check-order').click();
          await expect(page.locator('#order-details')).toContainText('1 at 101; fee 0.202 USDT');
          expect((await status()).operations).toHaveLength(1);
        }
        if(['paper','fill','protection_failed'].includes(scenario)) {
          if(scenario==='protection_failed') await fixture('reconcile');
          await refresh();
          await page.locator('[data-status-panel=positions] [data-close-symbol]').first().click();
          await expect(page.locator('#qtyInput')).toBeFocused();
          await page.locator('#qtyInput').fill('0.4');await review();
          if(scenario==='fill') await fixture('cancel-unknown');
          await confirm();await fixture('process');await page.locator('#check-order').click();
          if(scenario==='fill') {
            await expect(page.locator('#order-details')).toContainText('Cancellation pending');
            await expect(page.locator('#order-details')).toContainText('No confirmed fills');
            await expect(page.locator('#new-order')).toBeHidden();
            expect((await status()).fills.filter(fill=>fill.side==='sell')).toEqual([]);
          } else {
            await expect(page.locator('#order-details')).toContainText('0.4 at');
            expect((await status()).positions[0].qty).toBe('0.6');
          }
        }
        if(scenario==='paper') {
          await refresh();await page.locator('[data-status-panel=positions] [data-close-symbol]').first().click();
          await page.locator('#manual-order [name=sizing]').selectOption('full_close');await review();
          // Authentication interruption retains the signed close and inputs.
          const csrf=await page.locator('[name=csrf_token]').first().inputValue();
          await context.request.post(url+'/logout',{form:{csrf_token:csrf}});
          await page.locator('#confirm-order').click();
          await expect(page.locator('#order-message')).toContainText('Permission interrupted');
          const signIn=await context.newPage();await login(signIn);await signIn.close();await refresh();
          await confirm();await fixture('process');await page.locator('#check-order').click();
          expect((await status()).positions).toEqual([]);
          await refresh();
          await page.locator('[data-operation-id]').first().click();
          await expect(page.locator('#operation-history-detail')).toContainText('Confirmed fills: 1 at 101; fee 0 USDT');
          await expect(page.locator('#operation-history-detail')).toBeFocused();
          const before=(await status()).operations.length;
          await page.locator('#new-order').click();await page.locator('#qtyInput').fill('1');await review();
          await page.evaluate(()=> {
            window.savedStorageSetter=Storage.prototype.setItem;
            Storage.prototype.setItem=function(){throw new Error('Offline blocked browser storage');};
          });
          await page.locator('#confirm-order').click();
          await expect(page.locator('#order-message')).toContainText('Browser draft storage is unavailable');
          expect((await status()).operations).toHaveLength(before);
          await page.evaluate(()=>{Storage.prototype.setItem=window.savedStorageSetter;});
        }
        const axe=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
        expect(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))).toEqual([]);
        expect(errors).toEqual([]);
        const reflow=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
        expect(reflow).toBe(true);
        await page.locator('#order-review').scrollIntoViewIfNeeded();
        await page.screenshot({path:path.join(evidence,scenario+'.png')});
        results.push({scenario,passed:true,status:await status(),axeViolations:axe.violations.length,reflow});
        console.log('Passed browser recovery:',scenario);
      } finally {await context.close();server.kill();}
    }
    fs.writeFileSync(path.join(evidence,'journey.json'),JSON.stringify({revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
      scope:'Offline real Chromium and owned fake-provider lifecycle; no deployed or real-provider execution',
      browser:browser.version(),results,source_sha256:Object.fromEntries(['dashboard.py','state_store.py','manual_orders.py','templates/dashboard.html',
        'static/dashboard.js','static/manual-orders.js','static/dashboard.css'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(name)).digest('hex')]))},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
