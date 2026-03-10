# url_threat_detection/redirection_detector.py
import requests
import time
import re
import tldextract
import pandas as pd

from urllib.parse import urlparse

from validators import url
from .feature_extractor import safe_entropy
 
class URLRedirectionDetector:
    MAX_HOPS    = 5
    TIMEOUT     = 3
    THRESHOLD   = 0.7
    SHORTENERS  = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','short.link','buff.ly','adf.ly','rb.gy'}
    SUSPICIOUS_TLDS = {'tk','ml','ga','cf','gq','xyz','top','club','online','site','pw','cc'}
    PHISH_KEYWORDS  = ['login','signin','secure','account','verify','bank','paypal','password','confirm','wallet','reset']
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    def trace_redirections(self, url):
        chain = []
        current_url = url
        start_time  = time.time()
        for hop in range(self.MAX_HOPS):
            try:
                resp = requests.get(current_url, allow_redirects=False,
                                    timeout=self.TIMEOUT, headers=self.HEADERS)
                hop_time = round((time.time() - start_time) * 1000, 1)
                chain.append({'hop': hop+1, 'url': current_url,
                              'status_code': resp.status_code, 'response_ms': hop_time})
                if resp.status_code in (301, 302, 303, 307, 308):
                    next_url = resp.headers.get('Location', '')
                    if not next_url:
                        break
                    if next_url.startswith('/'):
                        p = urlparse(current_url)
                        next_url = f'{p.scheme}://{p.netloc}{next_url}'
                    if next_url == current_url:
                        chain[-1]['note'] = 'REDIRECT LOOP'
                        break
                    current_url = next_url
                else:
                    break
            except requests.exceptions.Timeout:
                chain.append({'hop': hop+1, 'url': current_url, 'error': 'TIMEOUT'})
                break
            except Exception as e:
                chain.append({'hop': hop+1, 'url': current_url, 'error': str(e)})
                break
        return {'chain': chain, 'total_hops': len(chain),
                'final_url': current_url, 'total_ms': round((time.time()-start_time)*1000, 1)}

    def compute_suspicion_score(self, original_url, trace_result, ml_prediction=None):
        score   = 0.0
        reasons = []
        chain   = trace_result['chain']
        hops    = trace_result['total_hops']
        final   = trace_result['final_url']

        if hops >= 5:
            score += 0.40; reasons.append(f'CRITICAL: {hops} hops (max reached)')
        elif hops >= 3:
            score += 0.20; reasons.append(f'WARNING: {hops} hops in chain')
        elif hops >= 2:
            score += 0.10; reasons.append(f'NOTE: {hops} hops in chain')

        for h in chain:
            ext = tldextract.extract(h.get('url', ''))
            dom = f'{ext.domain}.{ext.suffix}'
            if dom in self.SHORTENERS:
                score += 0.15
                reasons.append(f'Shortener at hop {h["hop"]}: {dom}')
                break

        orig_ext  = tldextract.extract(original_url)
        final_ext = tldextract.extract(final)
        orig_dom  = f'{orig_ext.domain}.{orig_ext.suffix}'
        final_dom = f'{final_ext.domain}.{final_ext.suffix}'
        if orig_dom != final_dom:
            score += 0.20
            reasons.append(f'Domain mismatch: {orig_dom} -> {final_dom}')

        if final_ext.suffix in self.SUSPICIOUS_TLDS:
            score += 0.15
            reasons.append(f'Suspicious TLD: .{final_ext.suffix}')

        ip_pattern = r'(?:\d{1,3}\.){3}\d{1,3}'
        if re.search(ip_pattern, final):
            score += 0.25; reasons.append('Final destination is raw IP address')

        kw_hits = [kw for kw in self.PHISH_KEYWORDS if kw in final]
        if kw_hits:
            score += min(0.15, 0.05 * len(kw_hits))
            reasons.append(f'Phishing keywords in final URL: {kw_hits}')

        ent = safe_entropy(final_ext.domain)
        if ent > 3.8:
            score += 0.15; reasons.append(f'High domain entropy ({ent:.2f}) - possible DGA')

        error_hops = [h for h in chain if 'error' in h]
        if error_hops:
            score += 0.10; reasons.append(f'{len(error_hops)} hop(s) had errors/timeouts')

        if ml_prediction is not None and ml_prediction != 0:
            lnames = {1:'phishing', 2:'malware', 3:'defacement'}
            score += 0.30
            reasons.append(f'ML model classified as: {lnames.get(ml_prediction, "malicious")}')

        score = min(1.0, round(score, 3))
        if score >= self.THRESHOLD:
            action = 'BLOCK'
        elif score >= 0.4:
            action = 'QUARANTINE'
        elif score >= 0.2:
            action = 'WARN'
        else:
            action = 'ALLOW'

        return {'suspicion_score': score, 'action': action,
                'is_malicious': score >= self.THRESHOLD,
                'reasons': reasons if reasons else ['No suspicious patterns detected']}

    def analyze_url(self, url, ml_model=None, feature_extractor=None):
        # ── ADD THESE 3 LINES at the top of the method ──────────────
        url       = str(url).strip()
        url_lower = url.lower()
        url       = url_lower if url_lower.startswith(('http://', 'https://')) else 'http://' + url_lower
        # (make the last 2 lines one line as shown in Step 4)
        # ─────────────────────────────────────────────────────────────
        # ── Normalise URL before any processing ──────────────────────────
        url = str(url).strip().lower()
        url = url if url.startswith(("http://","https://")) else "http://" + url
        # ─────────────────────────────────────────────────────────────────

        print(f"Analyzing: {url}")
        start = time.time()
        trace = self.trace_redirections(url)

        # ... rest of your existing code continues unchanged ...

        ml_pred = None
        if ml_model and feature_extractor:
            feats   = pd.DataFrame([feature_extractor(url)])
            ml_pred = int(ml_model.predict(feats)[0])
        score_result = self.compute_suspicion_score(url, trace, ml_pred)
        total_ms = round((time.time()-start)*1000, 1)
        report = {
            'original_url':    url,
            'trace':           trace,
            'ml_prediction':   {1:'phishing',2:'malware',3:'defacement',0:'benign'}.get(ml_pred,'unknown'),
            'suspicion_score': score_result['suspicion_score'],
            'action':          score_result['action'],
            'is_malicious':    score_result['is_malicious'],
            'reasons':         score_result['reasons'],
            'processing_ms':   total_ms
        }
        print(f'  Action: {report["action"]} | Score: {report["suspicion_score"]} | Hops: {trace["total_hops"]} | ML: {report["ml_prediction"]} | Time: {total_ms}ms')
        for r in report['reasons']:
            print(f'    * {r}')
        return report

