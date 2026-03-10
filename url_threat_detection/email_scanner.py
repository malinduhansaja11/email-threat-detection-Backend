# url_threat_detection/email_scanner.py
import re, email
from email import policy
from bs4 import BeautifulSoup
from .redirection_detector import URLRedirectionDetector
 
class EmailURLScanner:
    URL_REGEX = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

    def __init__(self, ml_model=None, feature_extractor=None):
        self.detector          = URLRedirectionDetector()
        self.ml_model          = ml_model
        self.feature_extractor = feature_extractor

    def extract_urls_from_email(self, raw_email_str):
        urls = set()
        msg = email.message_from_string(raw_email_str, policy=policy.default)
        for header in ['From', 'Reply-To', 'Return-Path']:
            urls.update(self.URL_REGEX.findall(msg.get(header, '')))
        for part in msg.walk():
            content_type = part.get_content_type()
            try:
                body = part.get_payload(decode=True)
                if body:
                    body = body.decode('utf-8', errors='ignore')
                    if content_type == 'text/html':
                        soup = BeautifulSoup(body, 'html.parser')
                        for tag in soup.find_all(['a','img','form','iframe']):
                            href = tag.get('href') or tag.get('src') or tag.get('action','')
                            if href and self.URL_REGEX.match(href):
                                urls.add(href)
                    urls.update(self.URL_REGEX.findall(body))
            except Exception:
                pass
        return list(urls)

    def scan_email(self, raw_email_str):
        urls = self.extract_urls_from_email(raw_email_str)
        print(f'Found {len(urls)} URL(s) in email')
        if not urls:
            return {'verdict': 'CLEAN', 'message': 'No URLs found', 'urls': []}
        results = []
        for url in urls:
            result = self.detector.analyze_url(url, ml_model=self.ml_model,
                                               feature_extractor=self.feature_extractor)
            results.append(result)
        blocked     = [r for r in results if r['action'] == 'BLOCK']
        quarantined = [r for r in results if r['action'] == 'QUARANTINE']
        warned      = [r for r in results if r['action'] == 'WARN']
        if blocked:
            verdict = 'BLOCK'
        elif quarantined:
            verdict = 'QUARANTINE'
        elif warned:
            verdict = 'WARN'
        else:
            verdict = 'ALLOW'
        max_score = max(r['suspicion_score'] for r in results)
        summary = {'verdict': verdict, 'max_score': max_score, 'total_urls': len(urls),
                   'blocked_urls': len(blocked), 'quarantine_urls': len(quarantined),
                   'warned_urls': len(warned), 'results': results}
        print(f'VERDICT: {verdict} | Max Score: {max_score} | Blocked: {len(blocked)} | Quarantined: {len(quarantined)}')
        return summary

