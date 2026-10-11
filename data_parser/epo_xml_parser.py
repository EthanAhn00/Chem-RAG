import os
import re

class EPOXMLParser:
    """EPO 특허 XML 파일에서 도면 태그와 figure 메타데이터를 확인하기 위한 파서
    <drawings> 블록을 추출하고 figure 이미지 존재 여부를 검증"""

    def __init__(self, xml_dir=None):
        if xml_dir is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            external_root = os.environ.get(
                'CHEMRAG_EXTERNAL_DATA_ROOT', os.path.join(project_dir, 'data', 'external')
            )
            xml_dir = os.environ.get(
                'CHEMRAG_EPO_XML_DIR', os.path.join(external_root, 'EPO_priorEPO_matched_xml_only')
            )
        self.xml_dir = xml_dir
        self.xml_map = {}
        self._build_xml_index()

    def _build_xml_index(self):
        # 파일명과 EP 문서번호(예: 'EP3315022') 둘 다로 조회 가능하게 인덱싱
        if not os.path.exists(self.xml_dir):
            return
        files = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]
        for f in files:
            self.xml_map[f] = f
            m = re.search(r'__(EP\d+)', f)
            if m:
                self.xml_map[m.group(1)] = f

    def inspect_drawing_metadata(self, filename_or_ep_id):
        xml_file = self.xml_map.get(filename_or_ep_id) or self.xml_map.get(filename_or_ep_id.replace('.json', '.xml'))

        if not xml_file:
            direct_path = os.path.join(self.xml_dir, filename_or_ep_id)
            if os.path.exists(direct_path):
                xml_file = filename_or_ep_id
            else:
                return {'has_drawing': False, 'figure_count': 0, 'figure_files': []}

        xml_path = os.path.join(self.xml_dir, xml_file)
        try:
            with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            fig_files = re.findall(r'<img[^>]+file="([^"]+)"[^>]*img-content="drawing"', text)
            if not fig_files:
                fig_files = re.findall(r'<figure[^>]*>.*?<img[^>]+file="([^"]+)"', text, re.DOTALL)

            return {
                'has_drawing': len(fig_files) > 0,
                'figure_count': len(fig_files),
                'figure_files': list(set(fig_files)),
                'xml_filename': xml_file
            }
        except Exception:
            return {'has_drawing': False, 'figure_count': 0, 'figure_files': []}
