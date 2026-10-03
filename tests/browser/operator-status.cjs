/* Actual Chromium journeys against the guarded loopback Flask/SQLite owner. */
const { chromium, expect } = require('@playwright/test');
const { default: AxeBuilder } = require('@axe-core/playwright');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync}=require('node:child_process');
const crypto=require('node:crypto');

(async () => {
  const state = fs.mkdtempSync(path.join(os.tmpdir(), 'bt12-browser-'));
  const evidence = path.resolve('docs/planning/trading-remediation/evidence/bt12-browser');
  fs.mkdirSync(evidence, { recursive: true });
  const server = spawn('python', ['-m', 'tests.operator_browser_server', state, '0'], { windowsHide: true });
  let output = '', browser;
  server.stdout.on('data', data => { output += data.toString(); });
  server.stderr.on('data', data => { output += data.toString(); });
  try {
    const start = Date.now();
    while (!output.match(/ready: (http:\/\/127\.0\.0\.1:\d+)/)) {
      if (server.exitCode !== null || Date.now() - start > 30000) throw new Error('Fixture startup failed: ' + output);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const url = output.match(/ready: (http:\/\/127\.0\.0\.1:\d+)/)[1];
    browser = await chromium.launch();
    const context = await browser.newContext({viewport:{width:1440,height:1000}, reducedMotion:'reduce'});
    await context.route('**/*', route => new URL(route.request().url()).origin === url ? route.continue() : route.abort());
    const page = await context.newPage();
    const failures = [];
    page.on('pageerror', failure => failures.push(failure.message));
    page.on('response', response => { if(response.status()>=400) console.log('HTTP fixture response',response.status(),new URL(response.url()).pathname); });
    async function signIn(target) {
      await target.goto(url + '/login');
      await target.locator('[name=username]').fill('operator');
      await target.locator('[name=password]').fill('offline-password');
      await target.locator('button[type=submit]').click();
      await target.waitForURL(url + '/');
      await expect(target.locator('#connection-status'),JSON.stringify(failures)).not.toHaveText('Status loaded.');
    }
    async function fixture(action) {
      const csrf = await page.locator('[name=csrf_token]').first().inputValue();
      const response = await context.request.post(url + '/__browser_fixture', {form:{action,csrf_token:csrf}});
      expect(response.status()).toBe(200);
    }
    async function refresh() {
      const reply = page.waitForResponse(response => response.url().includes('/api/status?view=1'));
      await page.locator('#refreshStatus').click(); await reply;
      await expect(page.locator('#connection-status')).not.toHaveAttribute('aria-busy','true');
    }
    await signIn(page);
    await expect(page.locator('[data-status-panel=performance]')).toContainText('Performance');
    await expect(page.locator('[data-status-panel=history]')).toContainText('Recent Trades');
    await expect(page.locator('[data-status-panel=health]')).toHaveText('Market data unavailable');
    await expect(page.locator('[data-status-panel=setup-status]')).toContainText('Paper setup required');
    await page.locator('#qtyInput').fill('0.2345');
    await page.locator('#settings-risk').fill('0.023');
    await page.locator('#strategy-profile').selectOption('daemon_daily');
    await page.locator('#settings-risk').focus();
    await page.waitForTimeout(16000); // One real production polling interval.
    await expect(page.locator('#settings-risk')).toBeFocused();
    await expect(page.locator('#settings-risk')).toHaveValue('0.023');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    await expect(page.locator('#strategy-profile')).toHaveValue('daemon_daily');
    const setupInputs={initial_cash:'10000',fee_fraction:'0',slippage_fraction:'0',risk_fraction:'1',fixed_quote:'100',
      max_allocation_fraction:'1',max_exposure_fraction:'1',max_positions:'100',cooldown_seconds:'0',drawdown_fraction:'1'};
    for (const [name,value] of Object.entries(setupInputs)) await page.locator('#setup-'+name).fill(value);
    await page.locator('form[action="/paper/setup"] button[type=submit]').click();
    await expect(page.locator('[data-status-panel=setup-status]')).toContainText('Explicit operator policy recorded.');
    await expect(page.locator('[data-status-panel=setup-status]')).not.toContainText('Paper setup required');
    await expect(page.locator('#settings-risk')).toHaveValue('0.023');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    await fixture('scan'); await refresh();
    await expect(page.locator('[data-status-panel=health]')).toContainText('Market data ready');
    const signal = await context.request.get(url + '/api/status');
    const original = await signal.json();
    expect(original.signals.length).toBeGreaterThan(0);
    const toast = await page.locator('#toast-root').textContent();
    expect(toast).toContain('New completed-candle entry signal');
    await page.evaluate(() => {
      window.alertChanges = 0;
      new MutationObserver(() => window.alertChanges++).observe(document.getElementById('toast-root'), {childList:true});
    });
    await fixture('scan'); await refresh();
    expect(await page.evaluate(() => window.alertChanges)).toBe(0);
    // Submit through authenticated HTTP and process with the real execution owner.
    async function paperOrder(side,qty) {
      const observed=await (await context.request.get(url+'/api/status')).json();
      const response=await context.request.post(url+'/trade',{form:{
        csrf_token:await page.locator('[name=csrf_token]').first().inputValue(),
        symbol:'BTC/USDT',side,qty,order_review:observed.order_review,
        idempotency_key:'browser-'+side, sizing:'quantity'}});
      expect(response.ok()).toBe(true); await fixture('process');
    }
    await paperOrder('buy','1'); await paperOrder('sell','0.4'); await refresh();
    const filled=await (await context.request.get(url+'/api/status')).json();
    expect(filled.fills.length).toBe(2);
    await expect(page.locator('[data-status-panel=history] tbody tr')).toHaveCount(2);
    await expect(page.locator('[data-status-panel=performance]')).toContainText('Complete');
    await page.locator('#settings-risk').focus();
    await fixture('stale');
    await page.waitForTimeout(16000);
    await expect(page.locator('[data-status-panel=performance]')).toContainText('Partial');
    await expect(page.locator('[data-status-panel=performance]')).toContainText('Estimated unrealized: Unavailable');
    await expect(page.locator('#settings-risk')).toBeFocused();
    await expect(page.locator('#settings-risk')).toHaveValue('0.023');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    await fixture('scan'); await refresh();

    await expect(page.locator('#soundToggle')).toHaveAttribute('aria-pressed','false');
    await page.locator('#soundToggle').click();
    await expect(page.locator('#soundToggle')).toHaveAttribute('aria-pressed','true');
    await fixture('stale'); await refresh();
    await page.evaluate(() => localStorage.removeItem('signal-candles:'+document.body.dataset.namespace));
    const stalePage=await context.newPage(); await stalePage.goto(url+'/');
    await expect(stalePage.locator('#connection-status')).toContainText('Market data degraded');
    await expect(stalePage.locator('#toast-root')).toBeEmpty();
    await stalePage.close();
    await fixture('empty'); await refresh();
    await expect(page.locator('#connection-status')).toHaveAttribute('data-state','no_signal');
    await fixture('stale'); await refresh();
    await expect(page.locator('[data-status-panel=health]')).toHaveText('Market data degraded');
    await expect(page.locator('[data-status-panel=context]')).toContainText('Data age: 61');
    await context.setOffline(true); await page.locator('#refreshStatus').click();
    await expect(page.locator('#connection-status')).toHaveAttribute('data-state','offline');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    await context.setOffline(false); await refresh();
    // Native valid values whose cross-field relationship must fail at server.
    await page.locator('#strategy-profile').selectOption('dashboard_intraday');
    await page.locator('#settings-rsi_exec_threshold').fill('80');
    await page.locator('#settings-rsi_overbought').fill('70');
    await page.locator('form[action="/update-config"] button[type=submit]').click();
    await expect(page.locator('#request-error')).toContainText('Execution RSI must be below');
    await expect(page.locator('#settings-rsi_exec_threshold')).toHaveValue('80');
    await expect(page.locator('#settings-rsi_exec_threshold')).toHaveAttribute('aria-invalid','true');
    await expect(page.locator('#request-error')).toBeFocused();
    await page.locator('#settings-rsi_exec_threshold').fill('30');
    const csrf = await page.locator('[name=csrf_token]').first().inputValue();
    expect((await context.request.post(url+'/update-config',{form:{csrf_token:csrf,risk:'0.04',expected_config_version:String(original.config_version)}})).status()).toBe(200);
    await page.locator('form[action="/update-config"] button[type=submit]').click();
    await expect(page.locator('#request-error')).toContainText('changed');
    await expect(page.locator('#settings-risk')).toHaveValue('0.023');
    await refresh();
    await page.locator('#reviewSettings').click();
    await expect(page.locator('#saved-settings')).toContainText('0.04');
    await page.locator('#acceptVersion').click();
    await expect(page.locator('#settings-risk')).toHaveValue('0.023');
    await page.locator('form[action="/update-config"] button[type=submit]').click();
    await expect(page.locator('#connection-status')).toContainText('Configuration saved.');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    await page.locator('#reviewSettings').click(); await page.locator('#acceptVersion').click();
    await page.locator('form[action="/reset-config"] button[type=submit]').click();
    await expect(page.locator('#connection-status')).toContainText('Configuration reset to defaults.');
    await expect(page.locator('#settings-risk')).toHaveValue('0.01');
    await expect(page.locator('#qtyInput')).toHaveValue('0.2345');
    const reset=await (await context.request.get(url+'/api/status')).json();
    expect(reset.execution_mode).toBe('paper'); expect(reset.namespace).toBe(original.namespace);
    expect(reset.config.paper_policy).toEqual(original.config.paper_policy);
    await page.locator('#reviewSettings').click(); await page.locator('#acceptVersion').click();
    await page.locator('#setup-fee_fraction').fill('1');
    await page.locator('form[action="/paper/setup"] button[type=submit]').click();
    await expect(page.locator('#setup-fee_fraction')).toHaveValue('1');
    await expect(page.locator('#setup-fee_fraction')).toHaveAttribute('aria-invalid','true');
    await expect(page.locator('#setup-error-fee_fraction')).toContainText('less than 1');
    await expect(page.locator('#settings-max_positions')).toHaveAttribute('aria-invalid','false');
    await page.locator('#setup-fee_fraction').fill('0');
    await page.locator('form[action="/paper/setup"] button[type=submit]').click();
    await expect(page.locator('#connection-status')).toContainText('Paper assumptions and explicit initial capital recorded');
    await page.locator('form[action="/logout"] button').click();
    // Return without navigation to preserve draft during auth interruption.
    await signIn(page);
    await page.locator('#qtyInput').fill('0.4567');
    const authPage = await context.newPage();
    await authPage.goto(url+'/'); await authPage.locator('form[action="/logout"] button').click();
    await refresh();
    await expect(page.locator('#connection-status')).toHaveAttribute('data-state','permission_limited');
    await expect(page.locator('#qtyInput')).toHaveValue('0.4567');
    await signIn(authPage); await refresh();
    await expect(page.locator('#qtyInput')).toHaveValue('0.4567');
    await authPage.close();
    const checks = [];
    for (const width of [1440,390,320]) {
      await page.setViewportSize({width,height:1000});
      await page.evaluate(() => scrollTo(0,0));
      const report = await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa','wcag22aa']).analyze();
      checks.push({width,violations:report.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>n.target)}))});
      expect(report.violations,JSON.stringify(checks)).toEqual([]);
      const reflow = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
      expect(reflow,'Page-level reflow at '+width).toBe(true);
      const smallTargets=await page.evaluate(() => Array.from(document.querySelectorAll('button,input:not([type=hidden]),select,a'))
        .filter(control=>control.getClientRects().length && !control.disabled)
        .filter(control=>{const rect=control.getBoundingClientRect();return rect.width<24 || rect.height<24;})
        .map(control=>control.outerHTML));
      expect(smallTargets,'Minimum24pixel targets at '+width).toEqual([]);
      await page.screenshot({path:path.join(evidence,'dashboard-'+width+'.png'),fullPage:true});
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(() => document.body.style.zoom='2');
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'200% zoom reflow').toBe(true);
    await page.screenshot({path:path.join(evidence,'dashboard-200-percent.png'),fullPage:true});
    await page.evaluate(() => document.body.style.zoom='1');
    await page.keyboard.press('Control+Home');
    await page.locator('.skip-link').focus(); await page.keyboard.press('Enter');
    await expect(page.locator('#main-content')).toBeFocused();
    await page.keyboard.press('Tab');
    const focus = await page.evaluate(() => {
      const style=getComputedStyle(document.activeElement), rect=document.activeElement.getBoundingClientRect();
      return {outline:style.outlineStyle,width:style.outlineWidth,visible:rect.top>=0&&rect.bottom<=innerHeight};
    });
    expect(focus.outline).toBe('solid'); expect(focus.visible).toBe(true);
    expect(failures).toEqual([]);
    const sourceFiles=['dashboard.py','templates/dashboard.html','static/dashboard.js','static/dashboard.css','assets/dashboard.css','package.json','package-lock.json'];
    fs.writeFileSync(path.join(evidence,'journey.json'),JSON.stringify({browser:await browser.version(),checks,
      source_revision:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
      source_sha256:Object.fromEntries(sourceFiles.map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(name)).digest('hex')])),
      polling_preserves_input_focus:true,offline_recovery:true,stale_data:true,auth_recovery:true,
      invalid_field_preservation:true,setup_field_recovery:true,initial_setup_banner_recovery:true,visible_performance_history_updates:true,version_conflict_review:true,candle_dedup:true,opt_in_sound:true,
      keyboard_focus:focus,css_scale_200_reflow:true,minimum_24_pixel_targets:true,page_errors:failures,
      boundary:'Actual Chromium, guarded loopback Flask/temp SQLite/owned fake public provider; no provider/deployed/device acceptance'},null,2));
    console.log('BT12 browser journeys passed:',JSON.stringify(checks));
  } finally {
    if (browser) await browser.close();
    server.kill();
    // Disposable financial state remains in the OS temporary directory.
  }
})().catch(error => { console.error(error); process.exitCode=1; });
