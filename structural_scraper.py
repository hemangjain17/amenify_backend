import json
import re
import urllib.parse
from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup, Tag

class SiteKnowledgeExtractor:
    """
    DOM -> Knowledge Transformer
    Structurally scrapes a webpage by parsing the DOM tree, removing noise,
    and extracting semantic blocks (sections, FAQs, lists, tables, links).
    """

    def __init__(self, user_agent: str = "AmenifyBot/1.0"):
        self.headers = {"User-Agent": user_agent}

    def fetch_html(self, url: str) -> str:
        """Fetch raw HTML from a URL."""
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response.text

    def get_dom(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def clean_dom(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove noisy elements like scripts, styles, headers, footers."""
        for tag in soup(["script", "style", "noscript", "footer", "nav", "header", "aside", "svg"]):
            tag.decompose()
        return soup

    def _get_heading_level(self, tag_name: str) -> int:
        if re.match(r'^h[1-6]$', tag_name):
            return int(tag_name[1])
        return 99

    def extract_hierarchy(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract sections based on heading hierarchy.
        Groups content under the most recent heading.
        """
        structured = []
        current_sections = {i: None for i in range(1, 7)} # Track current section at each heading level
        last_added_section = None

        # Find all relevant textual elements
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"]):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue

            if re.match(r'^h[1-6]$', tag.name):
                level = int(tag.name[1])
                section = {
                    "heading": text,
                    "level": tag.name,
                    "content": [],
                    "intent": self.tag_intent(text)
                }
                structured.append(section)
                
                # Update current section tracker
                current_sections[level] = section
                # Clear nested sections tracking
                for i in range(level + 1, 7):
                    current_sections[i] = None
                
                last_added_section = section

            elif last_added_section:
                last_added_section["content"].append(text)
            else:
                # Content before any heading
                pass

        # Cleanup empty sections
        return [s for s in structured if s["content"] or len(s["heading"].split()) > 3]

    def extract_faqs(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Detect structured FAQs based on question marks in headings or bold text
        and adjacent sibling content.
        """
        faqs = []
        for tag in soup.find_all(["h2", "h3", "h4", "strong", "button", "div"]):
            # Filter divs heavily to only those that look like accordion toggles
            if tag.name == "div" and not any(cls in " ".join(tag.get("class", [])).lower() for cls in ["faq", "question", "accordion", "toggle"]):
                continue

            question = tag.get_text(" ", strip=True)
            # A heuristic for a FAQ question: Ends with ? or contains typical question words
            is_question = "?" in question or re.match(r'^(how|what|why|when|where|do|can|is|are)\b', question.lower())
            
            if is_question and len(question) > 10 and len(question) < 200:
                answer_parts = []
                # Traverse siblings to collect answer
                for sib in tag.find_next_siblings():
                    if sib.name in ["h1", "h2", "h3", "h4"] or (sib.name == tag.name and ("?" in sib.get_text())):
                        break # Stop at next heading or question
                    
                    ans_text = sib.get_text(" ", strip=True)
                    if ans_text:
                        answer_parts.append(ans_text)
                
                if answer_parts:
                    faqs.append({
                        "question": question,
                        "answer": " \n".join(answer_parts)
                    })
        return faqs

    def extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract unordered and ordered lists (often features/steps)."""
        lists = []
        for ul in soup.find_all(["ul", "ol"]):
            items = []
            for li in ul.find_all("li", recursive=False): # Only direct children
                text = li.get_text(" ", strip=True)
                if text:
                    items.append(text)
            if items:
                lists.append({
                    "type": "ordered" if ul.name == "ol" else "unordered",
                    "items": items
                })
        return lists

    def extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract table data, row by row."""
        tables = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                if any(cols): # Only add row if it has some non-empty content
                    rows.append(cols)
            if rows:
                tables.append({
                    "type": "table",
                    "rows": rows
                })
        return tables

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all hyperlink structure and resolve relative URLs."""
        seen_urls = set()
        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            href = a["href"].strip()
            
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue

            # Resolve relative URLs (e.g. /blog/...) to absolute
            absolute_url = urllib.parse.urljoin(base_url, href)

            if absolute_url not in seen_urls and text:
                seen_urls.add(absolute_url)
                intent = "cta" if any(w in text.lower() for w in ["book", "sign up", "get started", "schedule", "contact", "apply"]) else "nav"
                links.append({
                    "text": text,
                    "url": absolute_url,
                    "intent": intent
                })
        return links

    def tag_intent(self, text: str) -> str:
        """Classify block intent based on text signals."""
        t = text.lower()
        if any(w in t for w in ["price", "cost", "fee", "plan", "$", "charge"]):
            return "pricing"
        elif any(w in t for w in ["service", "offer", "provide", "include"]):
            return "service"
        elif any(w in t for w in ["how", "faq", "question", "help", "support"]):
            return "faq"
        elif any(w in t for w in ["contact", "call", "email", "location"]):
            return "contact"
        return "general"

    def process_url(self, url: str) -> Dict[str, Any]:
        """Main pipeline to process a URL into structured JSON."""
        print(f"[Extractor] Fetching HTML from {url}...")
        try:
            html = self.fetch_html(url)
        except Exception as e:
            print(f"[Extractor] Failed to fetch {url}: {e}")
            return {"url": url, "error": str(e)}

        return self.process_html(html, url)

    def process_html(self, html: str, url: str) -> Dict[str, Any]:
        """Process raw HTML string into structured JSON."""
        print(f"[Extractor] Parsing DOM for {url}...")
        soup = self.get_dom(html)
        
        # 1. Clean noise
        soup = self.clean_dom(soup)

        # 2. Extract structural elements
        sections = self.extract_hierarchy(soup)
        faqs = self.extract_faqs(soup)
        lists = self.extract_lists(soup)
        tables = self.extract_tables(soup)
        links = self.extract_links(soup, base_url=url)

        structured_data = {
            "url": url,
            "sections": sections,
            "faqs": faqs,
            "lists": lists,
            "tables": tables,
            "links": links
        }
        
        print(f"[Extractor] Extracted {len(sections)} sections, {len(faqs)} FAQs, {len(lists)} lists, {len(tables)} tables, {len(links)} links.")
        return structured_data

if __name__ == "__main__":
    import sys
    import json
    
    AMENIFY_URLS = [
        "https://www.amenify.com",
        "https://www.amenify.com/resident-services",
        "https://www.amenify.com/cleaningservices1",
        "https://www.amenify.com/choreservices1",
        "https://www.amenify.com/handymanservices1",
        "https://www.amenify.com/professional-moving-services",
        "https://www.amenify.com/movingoutservices1",
        "https://www.amenify.com/groceryservices1",
        "https://www.amenify.com/dog-walking-services",
        "https://www.amenify.com/acommerce",
        "https://www.amenify.com/property-managers-2",
        "https://www.amenify.com/autogifts",
        "https://www.amenify.com/leasing-concession",
        "https://www.amenify.com/commercialcleaning1",
        "https://www.amenify.com/providers-1",
        "https://www.amenify.com/amenify-platform",
        "https://www.amenify.com/merchant-landing",
        "https://www.amenify.com/amenify-technology",
        "https://www.amenify.com/about-us",
        "https://www.amenify.com/news-articles",
        "https://www.amenify.com/blog",
        "https://www.amenify.com/contact-us",
        "https://www.amenify.com/resident-protection-plan",
        "https://www.amenify.com/faq",
        "https://www.amenify.com/sign-in-sign-up",
        "https://www.amenify.com/amenify-app",
        "https://www.amenify.com/tech-platform",
        "https://www.amenify.com/about",
        "https://www.amenify.com/services",
        "https://www.amenify.com/pricing",
        "https://www.amenify.com/careers",
        "https://www.amenify.com/technology",
        "https://www.amenify.com/cleaning-services",
        "https://www.amenify.com/chores-services",
        "https://www.amenify.com/handyman-services",
        "https://www.amenify.com/food-grocery-service",
        "https://www.amenify.com/move-out-cleaning-services",
    ]
    
    extractor = SiteKnowledgeExtractor()
    all_data = []
    
    print(f"Starting extraction for {len(AMENIFY_URLS)} URLs...")
    for url in AMENIFY_URLS:
        data = extractor.process_url(url)
        all_data.append(data)
    
    # Save all structural extractions to a single JSON KB
    output_file = "scraped_data_all.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[Success] Structured data for {len(AMENIFY_URLS)} pages saved to {output_file}")
