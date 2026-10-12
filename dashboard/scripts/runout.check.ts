// Runnable check: node --experimental-strip-types dashboard/scripts/runout.check.ts
import assert from 'node:assert/strict'
import { runsEmptyAt } from '../src/lib/runout.ts'

const h = 3600
// 1.0 -> 0.5 over two hours: 0.25 is reached one hour after the last capture
assert.equal(runsEmptyAt([{ ts: 0, fill: 1 }, { ts: h, fill: 0.75 }, { ts: 2 * h, fill: 0.5 }]), 3 * h)
// a restock at 2h starts a new run with too few points
assert.equal(runsEmptyAt([{ ts: 0, fill: 1 }, { ts: h, fill: 0.5 }, { ts: 2 * h, fill: 1 }, { ts: 3 * h, fill: 0.9 }]), null)
// flat, rising, already empty, too far ahead
assert.equal(runsEmptyAt([{ ts: 0, fill: 1 }, { ts: h, fill: 1 }, { ts: 2 * h, fill: 1 }]), null)
assert.equal(runsEmptyAt([{ ts: 0, fill: 0.5 }, { ts: h, fill: 0.6 }, { ts: 2 * h, fill: 0.7 }]), null)
assert.equal(runsEmptyAt([{ ts: 0, fill: 0.6 }, { ts: h, fill: 0.4 }, { ts: 2 * h, fill: 0.2 }]), null)
assert.equal(runsEmptyAt([{ ts: 0, fill: 1 }, { ts: 2 * h, fill: 0.9 }, { ts: 4 * h, fill: 0.8 }]), null) // 0.25 at 11.5 h, beyond 6 h
assert.equal(runsEmptyAt([{ ts: 0, fill: 0.6 }, { ts: h, fill: 0.58 }, { ts: 2 * h, fill: 0.56 }]), null) // noise-sized fall
// only the latest WINDOW captures set the rate: a flat morning does not slow down an evening sell-off
const evening = [1, 1, 1, 0.875, 0.75, 0.625].map((fill, i) => ({ ts: i * h, fill })) // last 4 fall 0.125 per hour
assert.equal(runsEmptyAt(evening), 8 * h)
console.log('runout ok')
