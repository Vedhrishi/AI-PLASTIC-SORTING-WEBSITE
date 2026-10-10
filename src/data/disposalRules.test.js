import { describe, expect, it } from 'vitest';
import { CONTAMINATION_ORDER, DISPOSAL_RULES, RESIN_ORDER, getDisposalRule } from './disposalRules';
import { RESIN_INFO } from './resinInfo';

describe('disposal matrix', () => {
  it('covers every resin x contamination level with a complete rule (12 cells)', () => {
    expect(Object.keys(DISPOSAL_RULES).sort()).toEqual([...RESIN_ORDER].sort());
    let cells = 0;
    for (const resin of RESIN_ORDER) {
      expect(Object.keys(DISPOSAL_RULES[resin]).sort()).toEqual([...CONTAMINATION_ORDER].sort());
      for (const level of CONTAMINATION_ORDER) {
        const rule = getDisposalRule(resin, level);
        expect(rule.action, `${resin}/${level} action`).toMatch(/\S/);
        expect(rule.route, `${resin}/${level} route`).toMatch(/\S/);
        expect(rule.reuse, `${resin}/${level} reuse`).toMatch(/\S/);
        cells += 1;
      }
    }
    expect(cells).toBe(12);
  });

  it('never recommends recycling for heavily contaminated items', () => {
    for (const resin of RESIN_ORDER) {
      expect(getDisposalRule(resin, 'Heavy Contamination').action).toMatch(/dispose/i);
    }
  });

  it('falls back to a manual-inspection rule for an unknown resin or level', () => {
    expect(getDisposalRule('PVC', 'Clean/Light Soiling').route).toMatch(/manual inspection/i);
    expect(getDisposalRule('PET', 'Radioactive').action).toBe('Unknown');
  });

  it('has resin facts for every resin in the matrix', () => {
    for (const resin of RESIN_ORDER) {
      expect(RESIN_INFO[resin]).toMatchObject({
        meltingPoint: expect.any(String),
        density: expect.any(String),
        viability: expect.any(String),
      });
    }
  });
});
