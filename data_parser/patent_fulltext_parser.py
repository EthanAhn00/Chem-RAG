"""Shared paragraph, figure, and claim-element parser for EPO ep-patent-document XML. Local XML (EPO_priorEPO_matched_xml_only) and OPS /description and /claims responses share this DTD, so both paths reuse this module."""
import re


def _strip_tags(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def parse_paragraphs(text):
    """Extract [{'para_no','text'}, ...] from <description>...<p id num>body</p>... ."""
    desc_m = re.search(r'<description.*?</description>', text, re.DOTALL)
    desc = desc_m.group(0) if desc_m else ''
    paragraphs = []
    for num, body in re.findall(r'<p id="[^"]+" num="(\d+)">(.*?)</p>', desc, re.DOTALL):
        clean = _strip_tags(body)
        if clean:
            paragraphs.append({'para_no': num, 'text': clean})
    return paragraphs


def parse_figures(text):
    """Match <figure id num>, captions from <li> sentences in 'Description of the Drawing', and <img> filenames. Return {fig_id: {'fig_no','caption','image_path'}}."""
    desc_m = re.search(r'<description.*?</description>', text, re.DOTALL)
    desc = desc_m.group(0) if desc_m else ''

    figures = {fid: {'fig_no': fnum, 'caption': None, 'image_path': None}
               for fid, fnum in re.findall(r'<figure id="([^"]+)" num="([^"]+)">', text)}

    # Assign the same <li> sentence as caption to every referenced <figref idref="...">.
    for li_body in re.findall(r'<li>(.*?)</li>', desc, re.DOTALL):
        caption_text = _strip_tags(li_body)
        for group in re.findall(r'<figref idref="([^"]+)"', li_body):
            for fid in group.split():
                if fid in figures and not figures[fid]['caption']:
                    figures[fid]['caption'] = caption_text

    # Conventional figure 'f0001' <-> image 'if0001' mapping; not guaranteed throughout the corpus.
    img_by_id = dict(re.findall(r'<img id="([^"]+)" file="([^"]+)"', text))
    for fid, meta in figures.items():
        meta['image_path'] = img_by_id.get('i' + fid)

    return figures


def parse_ops_paragraphs(text):
    """Extract paragraphs from OPS text-only /description responses. Unlike local <p id num> XML, numbering appears inside <p>[number] body</p>. Unnumbered headings such as 'Field of the Invention' retain para_no=None."""
    desc_m = re.search(r'<description[^>]*>(.*?)</description>', text, re.DOTALL)
    desc = desc_m.group(1) if desc_m else ''
    paragraphs = []
    for body in re.findall(r'<p>(.*?)</p>', desc, re.DOTALL):
        clean = _strip_tags(body)
        if not clean:
            continue
        num_m = re.match(r'^\[(\d+)\]\s*(.*)', clean)
        if num_m:
            paragraphs.append({'para_no': num_m.group(1), 'text': num_m.group(2).strip()})
        else:
            paragraphs.append({'para_no': None, 'text': clean})
    return paragraphs


def parse_ops_figures_heuristic(text):
    """OPS /description has no <figure> tags, unlike local XML. Approximate captions using sentences matching 'Fig. N ...'; image_path is always None because source drawing-file metadata is unavailable."""
    desc_m = re.search(r'<description[^>]*>(.*?)</description>', text, re.DOTALL)
    desc = desc_m.group(1) if desc_m else ''
    figures = {}
    for body in re.findall(r'<p>(.*?)</p>', desc, re.DOTALL):
        clean = re.sub(r'^\[\d+\]\s*', '', _strip_tags(body))
        for fig_m in re.finditer(r'\bFig(?:s|ure)?\.?\s*(\d+)[a-z]?', clean):
            fno = fig_m.group(1)
            fid = f'f{int(fno):04d}'
            if fid not in figures:
                figures[fid] = {'fig_no': fno, 'caption': clean, 'image_path': None}
    return figures


def parse_ops_claim_elements(text):
    """Extract claims from OPS text-only /claims. Unlike local <claim id num>, one <claim> contains multiple <claim-text> elements. Detect each claim start using leading '1. ', '2. ', etc.; attach subsequent limitation bullets to the same claim_no with sequential element_no. Return [(claim_no, element_no, text), ...], as in parse_claim_elements."""
    claim_m = re.search(r'<claim[^>]*>(.*?)</claim>', text, re.DOTALL)
    if not claim_m:
        return []
    blocks = re.findall(r'<claim-text>(.*?)</claim-text>', claim_m.group(1), re.DOTALL)

    # Further split limitations at '- ', matching split_target_claim_heuristic
    # so PRIOR and TARGET paths use comparable element granularity.
    claim_elements = []
    claim_no, element_no = None, 0
    for raw in blocks:
        clean = re.sub(r'\s+', ' ', _strip_tags(raw)).strip()
        if not clean:
            continue
        num_m = re.match(r'^(\d+)\.\s+(.*)', clean)
        if num_m:
            claim_no, element_no = num_m.group(1), 0
            clean = num_m.group(2)
        for sub in re.split(r';?\s+-\s+', clean):
            sub = sub.strip()
            if not sub:
                continue
            element_no += 1
            claim_elements.append((claim_no, element_no, sub))
    return claim_elements


def parse_claim_elements(text):
    """Each claim has nested <claim-text> tags: an outer full-text tag and inner limitations. Treat each tag's own text, before the next tag, as one element. Return [(claim_no, element_no, text), ...]."""
    claim_elements = []
    for claim_no, body in re.findall(r'<claim id="[^"]+" num="(\d+)">(.*?)</claim>', text, re.DOTALL):
        elements = [re.sub(r'\s+', ' ', m.group(1)).strip()
                    for m in re.finditer(r'<claim-text>([^<]*)', body)]
        for i, el_text in enumerate((e for e in elements if e), start=1):
            claim_elements.append((claim_no, i, el_text))
    return claim_elements
