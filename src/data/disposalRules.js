// Stage 3 disposal rules — localized to the Indian Plastic Waste Management
// (PWM) context: informal kabadiwala resale economics, mechanical/chemical
// recycling routes actually available domestically, and PWM-rule-driven
// end-of-life routing (cement-kiln co-processing rather than landfill for
// non-recyclables). Keyed the same way as before: resin -> contamination
// tier -> { action, route, reuse, note }.
export const DISPOSAL_RULES = {
  PET: {
    'Clean/Light Soiling': {
      action: 'Recycle - quick rinse first',
      route: 'Mechanical (bottle-to-bottle / fiber)',
      reuse: 'Rinsed planter, storage container',
      note: 'High kabadiwala demand, Rs 8-15/kg. Rinsed items fetch full value.',
    },
    'Moderate Contamination': {
      action: 'Recycle if cleaned; otherwise divert',
      route: 'Wash-then-mechanical; glycolysis chemical recycling where available',
      reuse: 'Not recommended (food-contact safety uncertain)',
      note: 'Informal-sector resale value drops sharply once soiled.',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable)',
      route: 'Cement-kiln co-processing (AFR) or road-construction bitumen mix',
      reuse: 'Not recommended',
      note: 'Not purchased by kabadiwala; PWM rules route non-recyclables to co-processing, not landfill.',
    },
  },
  HDPE: {
    'Clean/Light Soiling': {
      action: 'Recycle - quick rinse first',
      route: 'Mechanical (pipes/buckets/tiles)',
      reuse: 'Rinsed planter, storage bin',
      note: 'High kabadiwala demand, Rs 10-18/kg. Tolerates more handling than PET.',
    },
    'Moderate Contamination': {
      action: 'Recycle if cleaned; otherwise divert',
      route: 'Wash-then-mechanical; good pyrolysis feedstock',
      reuse: 'Not recommended',
      note: 'Value drops but less steeply than PET at this tier.',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable)',
      route: 'Cement-kiln co-processing or road-construction mix',
      reuse: 'Not recommended',
      note: 'Not purchased by kabadiwala at this tier.',
    },
  },
  PP: {
    'Clean/Light Soiling': {
      action: 'Recycle after quick rinse',
      route: 'Mechanical (storage bins, auto parts)',
      reuse: 'Microwave-safe storage box, bird feeder',
      note: 'Lower informal uptake than PET/HDPE; collection infrastructure thinner.',
    },
    'Moderate Contamination': {
      action: 'Prefer reuse over recycling',
      route: 'Wash-then-mechanical is economically marginal',
      reuse: 'Favor reuse suggestion over recycle',
      note: 'PP recycling is a 5-step process; contamination raises cost disproportionately.',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable)',
      route: 'Cement-kiln co-processing',
      reuse: 'Not recommended',
      note: 'Not purchased by kabadiwala at this tier.',
    },
  },
  PS: {
    'Clean/Light Soiling': {
      action: 'Recycle only if local buyer exists',
      route: 'Mechanical recycling rarely available',
      reuse: 'Non-food reuse only (storage)',
      note: 'EPS/foam and SUP-listed PS items banned in India since July 2022. Rigid PS rarely accepted by MRFs.',
    },
    'Moderate Contamination': {
      action: 'Dispose',
      route: 'Cement-kiln co-processing preferred over landfill',
      reuse: 'Not recommended',
      note: 'If item matches banned SUP-PS category, flag as banned.',
    },
    'Heavy Contamination': {
      action: 'Dispose',
      route: 'Cement-kiln co-processing or waste-to-energy',
      reuse: 'Not recommended',
      note: 'SUP-listed PS items banned in India since July 2022, regardless of contamination.',
    },
  },
};

export function getDisposalRule(resinLabel, contaminationLabel) {
  return (
    DISPOSAL_RULES[resinLabel]?.[contaminationLabel] ?? {
      action: 'Unknown',
      route: 'Manual inspection required',
      reuse: 'Unknown',
      note: '—',
    }
  );
}
