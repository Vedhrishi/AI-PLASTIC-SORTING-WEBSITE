import { chromium } from '@playwright/test';

const url = process.argv[2] || 'http://localhost:5177';
const out = process.argv[3] || 'screenshot.png';

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(1500);
await page.screenshot({ path: out, fullPage: true });

await browser.close();
console.log(`Saved ${out}`);
