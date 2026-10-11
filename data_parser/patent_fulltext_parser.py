"""EPO 특허문서(ep-patent-document DTD) 전문 XML에서 문단/도면/청구항 한정사항을
뽑는 공용 파서. 로컬 XML(EPO_priorEPO_matched_xml_only)과 OPS API(/description,
/claims) 응답이 같은 DTD를 쓰기 때문에 두 경로 모두 이 모듈을 재사용한다.
"""
import re


def _strip_tags(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def parse_paragraphs(text):
    """<description>...<p id num>본문</p>... 에서 [{'para_no','text'}, ...] 추출"""
    desc_m = re.search(r'<description.*?</description>', text, re.DOTALL)
    desc = desc_m.group(0) if desc_m else ''
    paragraphs = []
    for num, body in re.findall(r'<p id="[^"]+" num="(\d+)">(.*?)</p>', desc, re.DOTALL):
        clean = _strip_tags(body)
        if clean:
            paragraphs.append({'para_no': num, 'text': clean})
    return paragraphs


def parse_figures(text):
    """<figure id num> 목록 + "Description of the Drawing" 섹션의 <li> 문장에서
    캡션과 <img> 파일명을 매칭해 {fig_id: {'fig_no','caption','image_path'}} 반환"""
    desc_m = re.search(r'<description.*?</description>', text, re.DOTALL)
    desc = desc_m.group(0) if desc_m else ''

    figures = {fid: {'fig_no': fnum, 'caption': None, 'image_path': None}
               for fid, fnum in re.findall(r'<figure id="([^"]+)" num="([^"]+)">', text)}

    # 같은 <li> 안에 함께 나오는 모든 <figref idref="..."> 아이디에 그 문장을 캡션으로 부여
    for li_body in re.findall(r'<li>(.*?)</li>', desc, re.DOTALL):
        caption_text = _strip_tags(li_body)
        for group in re.findall(r'<figref idref="([^"]+)"', li_body):
            for fid in group.split():
                if fid in figures and not figures[fid]['caption']:
                    figures[fid]['caption'] = caption_text

    # figure id 'f0001' <-> img id 'if0001' 관례(전체 코퍼스에서 100% 보장되진 않음)
    img_by_id = dict(re.findall(r'<img id="([^"]+)" file="([^"]+)"', text))
    for fid, meta in figures.items():
        meta['image_path'] = img_by_id.get('i' + fid)

    return figures


def parse_ops_paragraphs(text):
    """OPS /description 응답(text-only 포맷)에서 문단 추출. 로컬 XML과 달리 <p id num>이
    아니라 그냥 <p>[번호]    본문</p> 형태라 번호가 텍스트 안에 대괄호로 박혀 있음.
    "Field of the Invention" 같은 소제목 문단은 번호가 없어 para_no=None으로 남김."""
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
    """OPS /description 응답엔 <figure> 태그 자체가 없어(로컬 XML과 결정적으로 다른 점)
    도면 목록을 구조적으로 뽑을 수 없음. 대신 본문에서 "Fig. N ..." 패턴이 나오는
    문장을 캡션 후보로 잡는 근사치 - image_path는 원본 도면 파일 정보가 아예 없어
    항상 None."""
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
    """OPS /claims 응답(text-only 포맷)에서 청구항 추출. 로컬 XML과 달리 <claim
    id num>으로 청구항이 나뉘어 있지 않고, 전체 청구항이 <claim> 태그 하나 안에
    <claim-text> 여러 개로 이어져 있음. claim_no는 각 청구항 첫 <claim-text> 맨 앞의
    "1. ", "2. " 같은 번호 패턴으로만 알 수 있어, 그 패턴이 나오면 새 청구항 시작으로
    보고 그 다음부터 나오는 <claim-text>들(한정사항 '- ' 불릿 등)을 같은 claim_no에
    element_no 순서대로 붙임. 반환 형식은 parse_claim_elements와 동일:
    [(claim_no, element_no, text), ...]"""
    claim_m = re.search(r'<claim[^>]*>(.*?)</claim>', text, re.DOTALL)
    if not claim_m:
        return []
    blocks = re.findall(r'<claim-text>(.*?)</claim-text>', claim_m.group(1), re.DOTALL)

    # "- " 구분자로 한정사항 단위까지 더 쪼갬 - TARGET 근사 분리(split_target_claim_heuristic)와
    # 같은 방식이라 PRIOR/TARGET 두 경로의 element 세분화 수준을 맞춤
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
    """청구항 하나당 <claim-text> 태그가 여러 겹 중첩됨(전문을 감싸는 최상위 1개 +
    한정사항마다 중첩된 것들). 태그 하나당 '자기 텍스트(다음 태그 전까지)'를 element
    하나로 잡아야 한정사항 단위로 쪼개짐. 반환: [(claim_no, element_no, text), ...]"""
    claim_elements = []
    for claim_no, body in re.findall(r'<claim id="[^"]+" num="(\d+)">(.*?)</claim>', text, re.DOTALL):
        elements = [re.sub(r'\s+', ' ', m.group(1)).strip()
                    for m in re.finditer(r'<claim-text>([^<]*)', body)]
        for i, el_text in enumerate((e for e in elements if e), start=1):
            claim_elements.append((claim_no, i, el_text))
    return claim_elements
