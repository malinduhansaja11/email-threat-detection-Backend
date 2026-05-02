# app/whitelist.py

# Common words whitelist
whitelist_words = set([
    "road", "years", "there", "are", "how", "to", "it", "i", "can", "use", "the",

    # extra safe/common words
    "account", "accounts", "service", "services", "anything", "changed",
    "change", "signin", "sign-in", "email", "security", "google",
    "recovery", "alert", "copy", "remove", "check", "activity",
    "important", "password", "welcome", "back", "secure", "new",
    "this", "that", "was", "need", "also", "received", "changes",
    "your", "you", "for", "help", "update"
])


whitelist_patterns = [

    # -----------------------------
    # Normal words
    # -----------------------------
    r'^[A-Za-z]+$',

    # normal words with ending punctuation
    r'^[A-Za-z]+\.$',
    r'^[A-Za-z]+,$',
    r'^[A-Za-z]+!$',
    r'^[A-Za-z]+\?$',
    r'^[A-Za-z]+:$',
    r'^[A-Za-z]+;$',

    # -----------------------------
    # Words with apostrophes
    # -----------------------------
    r"^[A-Za-z]+'[A-Za-z]+$",
    r"^[A-Za-z]+'[A-Za-z]+'[A-Za-z]+$",

    # apostrophe words with ending punctuation
    r"^[A-Za-z]+'[A-Za-z]+\.$",
    r"^[A-Za-z]+'[A-Za-z]+,$",
    r"^[A-Za-z]+'[A-Za-z]+!$",
    r"^[A-Za-z]+'[A-Za-z]+\?$",
    r"^[A-Za-z]+'[A-Za-z]+:$",
    r"^[A-Za-z]+'[A-Za-z]+;$",

    # -----------------------------
    # Hyphenated words
    # -----------------------------
    r'^[A-Za-z]+-[A-Za-z]+$',
    r'^[A-Za-z]+-[A-Za-z]+-[A-Za-z]+$',

    # hyphen words with ending punctuation
    r'^[A-Za-z]+-[A-Za-z]+\.$',
    r'^[A-Za-z]+-[A-Za-z]+,$',
    r'^[A-Za-z]+-[A-Za-z]+!$',
    r'^[A-Za-z]+-[A-Za-z]+\?$',
    r'^[A-Za-z]+-[A-Za-z]+:$',
    r'^[A-Za-z]+-[A-Za-z]+;$',

    # -----------------------------
    # Words inside quotes
    # -----------------------------
    r'^".*"$',
    r"^'.*'$",

    # -----------------------------
    # Chemical formulas
    # -----------------------------
   # r'^[A-Za-z][A-Za-z]?\d+[A-Za-z0-9]*$',   # H2O, CO2, co2
     r'^(?:[A-Z][a-z]?\d*){2,}$',
     r'^(?:co2|h2o|ch4|nh3|o2|n2)$',
    # -----------------------------
    # Dates
    # -----------------------------
    r'^\d{4}[.-]\d{1,2}[.-]\d{1,2}$',
    r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',

    # -----------------------------
    # Times
    # -----------------------------
    r'^\d{1,2}[:.]\d{2}$',
    r'^\d{1,2}(am|pm|AM|PM)$',

    # -----------------------------
    # Percentages
    # -----------------------------
    r'^\d+%$',

    # -----------------------------
    # Money formats
    # -----------------------------
    r'^\$\d+(\.\d+)?$',
    r'^\d+\$$',

    # -----------------------------
    # Time units
    # -----------------------------
    r'^\d+(s|m|h)$',

    # -----------------------------
    # Large numbers
    # -----------------------------
    r'^\d+(k|K|M|m|B|b)$',

    # -----------------------------
    # Measurements
    # -----------------------------
    r'^\d+(mm|cm|m|km|in|ft|yd|kg|g|mg|lb)$',

    # -----------------------------
    # Email addresses
    # -----------------------------
    #r'^[\w\.-]+@[\w\.-]+\.\w+\.<$',

    # email addresses with ending punctuation
    #r'^[\w\.-]+@[\w\.-]+\.\w+\.$',
   #r'^[\w\.-]+@[\w\.-]+\.\w+,$',
# normal email: user1592@gmail.com
r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$',

# email inside angle brackets: <user1592@gmail.com>
#r'^<[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>$',

# email with ending full stop: user1592@gmail.com.
r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\.$',

# email with ending comma: user1592@gmail.com,
r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,},$',

# email with ending colon/semicolon
r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}[:;]$',

# email inside angle brackets with punctuation: <user1592@gmail.com>,
r'^<[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>[.,;:]?$',
r'^(from|to|cc|bcc|subject|sent|date|reply-to):?$',

# full angle bracket email
r'^<[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>$',

# angle bracket email with punctuation
r'^<?[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>?[.,;:]?$',

# mailto format
r'^<?mailto:[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>?[.,;:]?$',
    # -----------------------------
    # Websites
    # -----------------------------
    r'^www\.[\w\.-]+\.\w+[/\w\.-]*$',
    r'^https?:\/\/[\w\.-]+\.\w+([\/\w\.-]*)?$',

    # websites with trailing punctuation
    r'^www\.[\w\.-]+\.\w+[/\w\.-]*\.$',
    r'^https?:\/\/[\w\.-]+\.\w+([\/\w\.-]*)?\.$',

    # -----------------------------
    # IP addresses
    # -----------------------------
    r'^\d{1,3}(\.\d{1,3}){3}$',

    # -----------------------------
    # Phone numbers
    # -----------------------------
    r'^\+?\d{7,15}$',

    # -----------------------------
    # Version numbers
    # -----------------------------
    r'^v?\d+(\.\d+)+$',

    # -----------------------------
    # Plain numbers
    # -----------------------------
    r'^\d+$',
    r'^\d+\.\d+$'
]