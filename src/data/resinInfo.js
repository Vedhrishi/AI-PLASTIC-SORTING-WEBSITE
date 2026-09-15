// Reference chemical/recycling data per resin. Static reference figures for
// display purposes (typical published ranges), not per-item lab measurements.
export const RESIN_INFO = {
  PET: {
    fullName: 'Polyethylene Terephthalate',
    meltingPoint: '250-260°C',
    density: '1.38-1.41 g/cm³',
    viability: 'High — widely accepted in curbside programs',
    pathways: {
      mechanical: 'Shredded, washed, and re-extruded into rPET flake for fiber, sheet, or new bottles.',
      pyrolysis: 'Not typically pyrolyzed — mechanical and glycolysis routes are far more efficient for PET.',
      glycolysis: 'Depolymerized with ethylene glycol back to monomers (BHET) for closed-loop virgin-grade PET.',
      wasteToEnergy: 'Last-resort incineration recovers ~23 MJ/kg; avoided where recycling infrastructure exists.',
    },
  },
  HDPE: {
    fullName: 'High-Density Polyethylene',
    meltingPoint: '120-140°C',
    density: '0.93-0.97 g/cm³',
    viability: 'High — one of the most recycled plastics by volume',
    pathways: {
      mechanical: 'Granulated and reprocessed into pipes, crates, and lumber-substitute products.',
      pyrolysis: 'Thermally cracked into naphtha/diesel-range hydrocarbons for chemical feedstock recovery.',
      glycolysis: 'Not applicable — polyolefins lack the ester linkages glycolysis targets.',
      wasteToEnergy: 'High calorific value (~46 MJ/kg) makes it a common feedstock when contamination blocks recycling.',
    },
  },
  PP: {
    fullName: 'Polypropylene',
    meltingPoint: '160-170°C',
    density: '0.90-0.91 g/cm³',
    viability: 'Moderate — accepted in a growing share of programs',
    pathways: {
      mechanical: 'Reground into pellets for automotive parts, containers, and textiles.',
      pyrolysis: 'Cracked into propylene-rich pyrolysis oil, usable as refinery feedstock.',
      glycolysis: 'Not applicable to polyolefins.',
      wasteToEnergy: 'Energy recovery (~44 MJ/kg) when mechanical recycling streams are unavailable.',
    },
  },
  PS: {
    fullName: 'Polystyrene',
    meltingPoint: '210-249°C',
    density: '1.04-1.09 g/cm³',
    viability: 'Low — rarely accepted curbside, needs specialty drop-off',
    pathways: {
      mechanical: 'Technically possible but rare — low bulk density makes collection/transport costly.',
      pyrolysis: 'Depolymerizes cleanly back toward styrene monomer, an emerging chemical-recycling route.',
      glycolysis: 'Not applicable — PS has no ester backbone.',
      wasteToEnergy: 'Common end-of-life route given limited mechanical/chemical recycling access (~41 MJ/kg).',
    },
  },
};

export const CONTAMINATION_INFO = {
  'Clean/Light Soiling': {
    index: 0,
    description: 'Minimal residue; suitable for standard recycling streams as-is or with a light rinse.',
  },
  'Moderate Contamination': {
    index: 1,
    description: 'Visible residue or labeling adhesive; requires a rinse cycle before entering the recycling stream.',
  },
  'Heavy Contamination': {
    index: 2,
    description: 'Significant organic or chemical residue; contaminates the recycling batch and is typically routed to general waste.',
  },
};
