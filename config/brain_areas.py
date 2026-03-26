"""
Brain area definitions and naming conventions
"""

# Brain area lists
BRAIN_AREAS = {
    'AV': ['AV'],
    'CL': ['CL'],
    'CP': ['CP'],
    'APN': ['APN'],
    'ProS': ['ProS'],
    'MB': ['MB'],
    'NOT': ['NOT'],
    'MD': ['MD'],
    'AMd': ['AMd'],
    'VISp': ['VISp'],
    'VISpm': ['VISpm'],
    'VISam': ['VISam'],
    'VISa': ['VISa'],
    'MOp': ['MOp'],
    'MOs': ['MOs'],
    'SUB': ['SUB'],
    'CA1': ['CA1'],
    'CA3': ['CA3'],
    'RSPagl': ['rsp_agl'],
    'RSPd': ['rsp_d'],
    'RSPv': ['rsp_v'],
    'DG_po': ['DG-po'],
    'DG_sg': ['DG-sg'],
    'DG_mo': ['DG-mo'],
    'SSp_ll': ['SSp_ll'],
    'ACAd': ['ACAd'],
    'LP': ['vis_thalamus']
}

# Flattened list of all brain areas
ALL_BRAIN_AREAS = [area for sublist in BRAIN_AREAS.values() for area in sublist]

# Brain area name mapping (internal -> display)
AREA_NAME_MAPPING = {
    'rsp_v': 'RSPv',
    'vis_thalamus': 'LP',
    'rsp_d': 'RSPd',
    'SSp_ll': 'SSp-ll',
    'rsp_agl': 'RSPagl',
    'visp': 'VISp',
    'mop': 'MOp',
    'mos': 'MOs',
    'ss_ctx': 'SSp-tr',
    'anterior_cingulate': 'SSp-tr'
}

def get_display_name(internal_name):
    """Convert internal brain area name to display name"""
    return AREA_NAME_MAPPING.get(internal_name, internal_name)
