// Stage 3 disposal rule table: (resin, contamination) -> action / route / reuse.
export const DISPOSAL_RULES = {
  PET: {
    'Clean/Light Soiling': {
      action: 'Recycle',
      route: 'Curbside recycling (resin code 1)',
      reuse: 'Rinse and reuse as a water bottle or storage container',
    },
    'Moderate Contamination': {
      action: 'Rinse, then recycle',
      route: 'Curbside recycling after removing residue',
      reuse: 'Not recommended for reuse — rinse thoroughly first',
    },
    'Heavy Contamination': {
      action: 'Discard',
      route: 'General waste (residue prevents clean recycling)',
      reuse: 'Not suitable for reuse',
    },
  },
  HDPE: {
    'Clean/Light Soiling': {
      action: 'Recycle',
      route: 'Curbside recycling (resin code 2)',
      reuse: 'Reuse as storage container or planter',
    },
    'Moderate Contamination': {
      action: 'Rinse, then recycle',
      route: 'Curbside recycling after removing residue',
      reuse: 'Rinse well before any reuse',
    },
    'Heavy Contamination': {
      action: 'Discard',
      route: 'General waste',
      reuse: 'Not suitable for reuse',
    },
  },
  PP: {
    'Clean/Light Soiling': {
      action: 'Recycle',
      route: 'Curbside recycling (resin code 5, check local rules)',
      reuse: 'Reuse as food storage container (microwave-safe)',
    },
    'Moderate Contamination': {
      action: 'Rinse, then recycle',
      route: 'Curbside recycling after removing residue',
      reuse: 'Not recommended for reuse until fully cleaned',
    },
    'Heavy Contamination': {
      action: 'Discard',
      route: 'General waste',
      reuse: 'Not suitable for reuse',
    },
  },
  PS: {
    'Clean/Light Soiling': {
      action: 'Special drop-off',
      route: 'PS/Styrofoam is rarely curbside-recyclable — find a specialty drop-off',
      reuse: 'Reuse for shipping/packing cushioning',
    },
    'Moderate Contamination': {
      action: 'Discard',
      route: 'General waste (PS recycling infrastructure is limited)',
      reuse: 'Not suitable for reuse',
    },
    'Heavy Contamination': {
      action: 'Discard',
      route: 'General waste',
      reuse: 'Not suitable for reuse',
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
