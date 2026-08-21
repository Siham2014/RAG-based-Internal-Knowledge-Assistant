from __future__ import annotations

import re


class FollowUpDetector:
    """Conservative, domain-independent detector for context-dependent turns."""

    _DEPENDENT_START = re.compile(
        r"^(?:and|also|but|why|how|what about|more|explain more|"
        r"et|mais|aussi|pourquoi|comment|et concernant|explique|plus de|"
        r"وماذا|ماذا عن|ولماذا|لماذا|كيف|اشرح|المزيد)\b",
        re.IGNORECASE,
    )
    _REFERENCE = re.compile(
        r"\b(?:it|this|that|these|those|them|its|"
        r"cela|ça|ceci|celui|celle|eux|"
        r"هذا|هذه|ذلك|تلك|عنها|عنه)\b",
        re.IGNORECASE,
    )

    def is_follow_up(self, message: str, has_history: bool) -> bool:
        if not has_history:
            return False
        text = str(message or "").strip()
        if not text:
            return False
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        return bool(
            self._DEPENDENT_START.search(text)
            or self._REFERENCE.search(text)
            or (len(tokens) <= 2 and text.endswith("?"))
        )

