// Stage 3 disposal rules — Indian Plastic Waste Management (PWM) context.
// Keyed the same way throughout the app: resin -> contamination tier ->
// { action, route, reuse }. Resin keys use the app's canonical labels
// (HDPE, matching classify.js's RESIN_CODES output) rather than the ISO
// "PE-HD" designation from the source dataset.
export const DISPOSAL_RULES = {
  PET: {
    'Clean/Light Soiling': {
      action: 'Recycle - give it a quick rinse first if any residue is visible',
      route: 'Mechanical (bottle-to-bottle / fiber)',
      reuse: 'Rinsed planter, storage container, craft/DIY reuse',
    },
    'Moderate Contamination': {
      action: 'Recycle if cleaned; otherwise divert',
      route: 'Wash-then-mechanical; PET is a poor pyrolysis feedstock so chemical recycling route is glycolysis',
      reuse: 'Not recommended (food-contact safety uncertain once soiled)',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable at this tier)',
      route: 'Cement-kiln co-processing (AFR) or shredded into road-construction bitumen mix',
      reuse: 'Not recommended',
    },
  },
  HDPE: {
    'Clean/Light Soiling': {
      action: 'Recycle - give it a quick rinse first if any residue is visible',
      route: 'Mechanical (bottle-to-bottle, or downcycled to pipes/buckets/tiles)',
      reuse: 'Rinsed planter, storage bin, bird feeder',
    },
    'Moderate Contamination': {
      action: 'Recycle if cleaned; otherwise divert',
      route: 'Wash-then-mechanical; good pyrolysis feedstock (olefin) where capacity exists',
      reuse: 'Not recommended',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable at this tier)',
      route: 'Cement-kiln co-processing or road-construction mix',
      reuse: 'Not recommended',
    },
  },
  PP: {
    'Clean/Light Soiling': {
      action: 'Recycle after a quick rinse where collection exists, else prefer reuse',
      route: 'Mechanical (storage bins, auto parts, pallets)',
      reuse: 'Microwave-safe storage box, bird feeder (sturdy, reusable)',
    },
    'Moderate Contamination': {
      action: 'Prefer reuse over recycling; dispose if neither applies',
      route: 'Wash-then-mechanical is economically marginal at this tier',
      reuse: 'Favor reuse suggestion over recycle at this tier',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable at this tier)',
      route: 'Cement-kiln co-processing or road-construction mix',
      reuse: 'Not recommended',
    },
  },
  PS: {
    'Clean/Light Soiling': {
      action: 'Recycle only if a regional buyer exists, else reuse cautiously or dispose',
      route: 'Mechanical recycling technically possible but rarely available in practice',
      reuse: 'Non-food reuse only (storage) - avoid food reuse given brittleness',
    },
    'Moderate Contamination': {
      action: 'Dispose (recycling unlikely to find a buyer at this tier)',
      route: 'Cement-kiln co-processing preferred over landfill',
      reuse: 'Not recommended',
    },
    'Heavy Contamination': {
      action: 'Dispose (non-recyclable)',
      route: 'Cement-kiln co-processing or waste-to-energy',
      reuse: 'Not recommended',
    },
  },
};

export function getDisposalRule(resinLabel, contaminationLabel) {
  return (
    DISPOSAL_RULES[resinLabel]?.[contaminationLabel] ?? {
      action: 'Unknown',
      route: 'Manual inspection required',
      reuse: 'Unknown',
    }
  );
}
