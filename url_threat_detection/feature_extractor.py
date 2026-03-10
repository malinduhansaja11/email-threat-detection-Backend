# url_threat_detection/feature_extractor.py
import re, math, tldextract
from urllib.parse import urlparse
from collections import Counter
 
TRUSTED_DOMAINS = {
    'google.com', 'github.com', 'youtube.com', 'facebook.com',
    'twitter.com', 'linkedin.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'stackoverflow.com', 'wikipedia.org',
    'instagram.com', 'reddit.com', 'netflix.com', 'spotify.com'
}
 
BRANDS = ['paypal','google','facebook','microsoft','apple',
          'amazon','netflix','instagram','twitter','linkedin',
          'chase','wellsfargo','bankofamerica','citibank']
 
def safe_entropy(s):
    if not s or len(s) == 0:
        return 0.0
    probs = [n / len(s) for n in Counter(s).values()]
    return round(-sum(p * math.log2(p) for p in probs), 4)
 
def extract_features(url):
    features = {}
    try:
        url = str(url).lower().strip()
        url_for_parse = url if url.startswith(("http://", "https://")) else "http://" + url   
        parsed = urlparse(url_for_parse)       
        ext    = tldextract.extract(url_for_parse)

        try:
            parsed = urlparse(url_for_parse)
            _ = parsed.netloc
        except Exception:
            parsed = urlparse('http://invalid.com')

        try:
            ext = tldextract.extract(url_for_parse)
        except Exception:
            ext = tldextract.extract('http://invalid.com')

        domain    = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
        subdomain = ext.subdomain
        url_lower = url

        # ── Original features ────────────────────────────────────────────────
        features['url_length']            = len(url)
        features['domain_length']         = len(domain)
        features['path_length']           = len(parsed.path)
        features['query_length']          = len(parsed.query)
        features['subdomain_length']      = len(subdomain)
        features['dot_count']             = url.count('.')
        features['slash_count']           = url.count('/')
        features['hyphen_count']          = url.count('-')
        features['at_count']              = url.count('@')
        features['question_count']        = url.count('?')
        features['amp_count']             = url.count('&')
        features['equal_count']           = url.count('=')
        features['underscore_count']      = url.count('_')
        features['percent_count']         = url.count('%')
        features['digit_count']           = sum(c.isdigit() for c in url)
        features['letter_count']          = sum(c.isalpha() for c in url)
        features['special_char_count']    = sum(not c.isalnum() and c not in '/:.-_?' for c in url)
        features['digit_letter_ratio']    = features['digit_count'] / max(features['letter_count'], 1)
        features['url_entropy']           = safe_entropy(url)
        features['domain_entropy']        = safe_entropy(ext.domain)
        features['path_entropy']          = safe_entropy(parsed.path)
        features['subdomain_count']       = len(subdomain.split('.')) if subdomain else 0
        features['path_depth']            = len([p for p in parsed.path.split('/') if p])
        features['has_port']              = 1 if parsed.port else 0
        features['has_query']             = 1 if parsed.query else 0
        features['has_fragment']          = 1 if parsed.fragment else 0
        features['is_https']              = 1 if parsed.scheme == 'https' else 0

        ip_pattern = r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
        features['has_ip']                = 1 if re.search(ip_pattern, url) else 0

        shorteners = ['bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','short.link','buff.ly','adf.ly','rb.gy']
        features['is_shortened']          = 1 if any(s in url for s in shorteners) else 0

        phish_kw = ['login','signin','secure','account','update','verify','bank','paypal',
                    'password','credential','confirm','wallet','support','alert','reset','unlock']
        features['phish_keyword_count']   = sum(kw in url for kw in phish_kw)

        mal_kw = ['download','install','setup','exe','dll','payload','dropper','exploit','shell','bot']
        features['malware_keyword_count'] = sum(kw in url for kw in mal_kw)

        sus_tlds = ['tk','ml','ga','cf','gq','xyz','top','club','online','site','pw','cc','info','biz']
        features['has_suspicious_tld']    = 1 if ext.suffix in sus_tlds else 0
        features['has_hex_encoding']      = 1 if re.search(r'%[0-9a-f]{2}', url) else 0
        features['has_double_slash']      = 1 if '//' in parsed.path else 0
        features['consecutive_digits']    = len(re.findall(r'\d{5,}', domain))

        # ── NEW FEATURE 1: Trusted domain ────────────────────────────────────
        actual_domain = ext.domain + '.' + ext.suffix
        features['is_trusted_domain']     = 1 if actual_domain in TRUSTED_DOMAINS else 0

        # ── NEW FEATURE 2: Brand name in subdomain (paypal.evil.com) ─────────
        features['has_brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in BRANDS) else 0

        # ── NEW FEATURE 3: Digits in domain (paypa1, g00gle) ─────────────────
        features['has_digit_in_domain']   = 1 if any(c.isdigit() for c in ext.domain) else 0

        # ── NEW FEATURE 4: Subdomain depth (a.b.c.d.evil.com) ────────────────
        features['subdomain_depth']       = len(subdomain.split('.')) if subdomain else 0

        # ── NEW FEATURE 5: Unusually long domain ─────────────────────────────
        features['domain_too_long']       = 1 if len(ext.domain) > 20 else 0

        # ── NEW FEATURE 6: Brand name + suspicious TLD ───────────────────────
        features['brand_with_sus_tld']    = 1 if (
            any(b in url_lower for b in BRANDS) and ext.suffix in sus_tlds
        ) else 0

        # ── NEW FEATURE 7: Punycode / homograph attack ────────────────────────
        features['has_punycode']          = 1 if 'xn--' in url_lower else 0

        # ── NEW FEATURE 8: Sensitive file extension in path ───────────────────
        sensitive_ext = ['.exe','.dll','.bat','.ps1','.vbs','.zip','.rar','.msi']
        features['has_sensitive_extension'] = 1 if any(e in parsed.path.lower() for e in sensitive_ext) else 0

        # ── NEW FEATURE 9: Too many hyphens in domain (secure-login-verify.com)
        features['domain_hyphen_count']   = ext.domain.count('-')

        # ── NEW FEATURE 10: URL has both @ and redirect chars ─────────────────
        features['has_at_redirect']       = 1 if '@' in url and ('/' in url.split('@')[-1]) else 0

    except Exception:
        default_keys = [
            'url_length','domain_length','path_length','query_length','subdomain_length',
            'dot_count','slash_count','hyphen_count','at_count','question_count',
            'amp_count','equal_count','underscore_count','percent_count',
            'digit_count','letter_count','special_char_count','digit_letter_ratio',
            'url_entropy','domain_entropy','path_entropy','subdomain_count',
            'path_depth','has_port','has_query','has_fragment','is_https',
            'has_ip','is_shortened','phish_keyword_count','malware_keyword_count',
            'has_suspicious_tld','has_hex_encoding','has_double_slash','consecutive_digits',
            # New features
            'is_trusted_domain','has_brand_in_subdomain','has_digit_in_domain',
            'subdomain_depth','domain_too_long','brand_with_sus_tld','has_punycode',
            'has_sensitive_extension','domain_hyphen_count','has_at_redirect'
        ]
        features = {k: 0 for k in default_keys}

    return features

