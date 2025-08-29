from rapidfuzz import process, fuzz
import re


class CategoryUtils:
    primary_categories = [
        "Foundations and Logic",
        "Algebra and Number Theory",
        "Analysis and Differential Equations",
        "Geometry and Topology",
        "Probability, Statistics, and Discrete Mathematics",
    "Applied and Computational Mathematics",
    "Arithmetic",
]
    secondary_categories = {
    "Foundations and Logic": [
        "Mathematical Logic and Set Theory",
        "Basic Theory, Formalization, and History & Education",
    ],
    "Algebra and Number Theory": [
        "Linear Algebra and Group Theory",
        "Ring Theory, Field Theory, and Polynomial Algebra",
        "Commutative Algebra and Homological/Categorical Methods",
        "Number Theory",
        "Algebraic Geometry",
    ],
    "Analysis and Differential Equations": [
        "Real Analysis, Measure Theory, and Functional Analysis",
        "Complex Analysis and Special Functions",
        "Differential Equations and Dynamical Systems",
        "Integral Transforms, Integral Equations, and Difference Equations",
        "Harmonic Analysis",
    ],
    "Geometry and Topology": [
        "Euclidean, Analytic, and Convex/Discrete Geometry",
        "Differential Geometry and Manifold Theory",
        "Topology and Algebraic Topology",
    ],
    "Probability, Statistics, and Discrete Mathematics": [
        "Probability Theory and Stochastic Processes",
        "Mathematical Statistics",
        "Combinatorics and Graph Theory",
    ],
    "Applied and Computational Mathematics": [
        "Numerical Analysis and Computational Methods",
        "Optimal Control, Variational Methods, and Optimization",
        "Operations Research and Game Theory",
        "Systems Theory and Control",
        "Computer Science and Algorithms",
        "Mathematical Physics and Engineering Mathematics",
        "Information and Communication",
        "Bimathematics",
    ],
    "Arithmetic": [
        "Basic Arithmetic and Number Operations",
        "Word Problems and Real-Life Applications",
    ],
}

    def normalize_text(self,s: str) -> str:
        s = s.lower()
        # 去数字、点、连字符、下划线、逗号等
        s = re.sub(r"[0-9\.\-\_\(\)\[\],&/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def fuzzy_match_label(self,
                        raw_label: str,
                        choices: list[str],
                        scorer=fuzz.WRatio,
                        threshold: int = 70) -> str | None:
        raw_clean = self.normalize_text(raw_label)
        best, score, _ = process.extractOne(
            query=raw_clean,
            choices=choices,
            scorer=scorer
    )
        return best if score >= threshold else None

    def normalize_categories(
        self,
        raw_primary: str,
        raw_secondary: str,
        primary_choices: list[str] = primary_categories,
        secondary_map: dict[str, list[str]] = secondary_categories,
        thresh_primary: int = 50,
        thresh_secondary: int = 50
    ) -> dict:
        """
        优先解析纯数字编号（如"1"、"1."或"1.1"），否则再走 fuzzy-match。
        返回 {'primary_category': ..., 'secondary_category': ...}。
        """
        # 1) 尝试直接按 "X.Y" 解析子类编号
        m = re.match(r"^\s*(\d+)\s*\.\s*(\d+)\s*\.?\s*$", raw_secondary)
        if m:
            pi, si = int(m.group(1)), int(m.group(2))
            if 1 <= pi <= len(primary_choices):
                primary = primary_choices[pi - 1]
                secs = secondary_map.get(primary, [])
                if 1 <= si <= len(secs):
                    return {
                        "primary_category": primary,
                        "secondary_category": secs[si - 1]
                    }
        # 2) 再尝试按 "X" 解析主类编号
        m = re.match(r"^\s*(\d+)\s*\.?\s*$", raw_primary)
        if m:
            pi = int(m.group(1))
            if 1 <= pi <= len(primary_choices):
                primary = primary_choices[pi - 1]
            else:
                primary = None
        else:
            primary = None

        # 3) 如果编号解析不到，再 fuzzy-match 主类
        if primary is None:
            primary = self.fuzzy_match_label(raw_primary, primary_choices, threshold=thresh_primary)

        # 如果主类依然没匹配上，直接返回 ""
        if not primary:
            return {"primary_category": "", "secondary_category": ""}

        # 4) 在该主类对应的二级列表中做 fuzzy-match
        sec_choices = secondary_map.get(primary, [])
        secondary = self.fuzzy_match_label(raw_secondary, sec_choices, threshold=thresh_secondary)
        if not secondary:
            secondary = ""

        return {
            "primary_category": primary,
            "secondary_category": secondary
        }

    def category_hasher(self,primary: str, secondary: str) -> float:
        # 对于第k个大类的第m个二级类，返回k * 8 + m
        try:
            k = self.primary_categories.index(primary)
            m = self.secondary_categories[primary].index(secondary)
            return k * 8 + m
        except:
            return 170

    def category_hasher_reverse(self,hash: float) -> tuple[str, str]:
        if hash < 0 or hash > 56:
            return None, None
        k = int(hash / 8)
        m = int(hash % 8)
        return self.primary_categories[k], self.secondary_categories[self.primary_categories[k]][m]


