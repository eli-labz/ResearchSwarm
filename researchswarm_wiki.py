"""ResearchSwarm markdown knowledge layer (LLM Wiki).

This module adds a persistent markdown wiki beside the SQLite memory ledger.
It supports three core operations: ingest, query, and lint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


@dataclass
class WikiQueryResult:
    path: Path
    score: int
    snippet: str


class ResearchSwarmWiki:
    CATEGORY_DIRS = {
        "experiments": "experiments",
        "architectures": "architectures",
        "optimizers": "optimizers",
        "datasets": "datasets",
        "failures": "failures",
        "hypotheses": "hypotheses",
    }

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parent
        self.wiki_root = self.repo_root / "LLM-Wiki"
        self.raw_root = self.wiki_root / "raw"
        self.wiki_pages_root = self.wiki_root / "wiki"
        self.index_path = self.wiki_pages_root / "index.md"
        self.log_path = self.wiki_pages_root / "log.md"
        self.schema_path = self.wiki_root / "AGENTS.md"

    def init(self) -> None:
        for subdir in ("papers", "run_logs", "experiment_notes", "external_sources"):
            (self.raw_root / subdir).mkdir(parents=True, exist_ok=True)

        for category in self.CATEGORY_DIRS.values():
            (self.wiki_pages_root / category).mkdir(parents=True, exist_ok=True)

        if not self.index_path.exists():
            self.index_path.write_text(
                "\n".join(
                    [
                        "# ResearchSwarm LLM Wiki Index",
                        "",
                        "This index tracks synthesized wiki pages generated from curated raw sources.",
                        "",
                        "## Experiments",
                        "",
                        "## Architectures",
                        "",
                        "## Optimizers",
                        "",
                        "## Datasets",
                        "",
                        "## Failures",
                        "",
                        "## Hypotheses",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

        if not self.log_path.exists():
            self.log_path.write_text(
                "# ResearchSwarm LLM Wiki Log\n\nAppend-only timeline of ingest, query, and lint operations.\n",
                encoding="utf-8",
            )

        if not self.schema_path.exists():
            self.schema_path.write_text(
                "\n".join(
                    [
                        "# ResearchSwarm LLM Wiki Schema",
                        "",
                        "This schema defines how agents maintain the LLM-Wiki knowledge layer.",
                        "",
                        "## Mission",
                        "- Keep AI-Memory/memory.db as the event ledger.",
                        "- Maintain LLM-Wiki/wiki as synthesized, explainable markdown memory.",
                        "- Never edit files under LLM-Wiki/raw because they are immutable sources.",
                    ]
                ),
                encoding="utf-8",
            )

    def ingest(self, source_path: str) -> str:
        self.init()
        if not source_path.strip():
            return "Wiki ingest failed: no source path provided."

        resolved = self._resolve_source_path(source_path)
        if resolved.exists() and resolved.is_dir():
            return self.ingest_batch(str(resolved))

        if not resolved.exists() or not resolved.is_file():
            return f"Wiki ingest failed: source file not found: {resolved}"

        text = resolved.read_text(encoding="utf-8", errors="replace")
        category = self._infer_category(text, resolved)
        page_slug = self._build_page_slug(resolved, text)
        page_path = self.wiki_pages_root / self.CATEGORY_DIRS[category] / f"{page_slug}.md"

        summary = self._summarize_text(text)
        evidence = self._extract_evidence_lines(text)
        source_rel = self._to_repo_relative(resolved)
        now = self._timestamp()
        related_pages = self._find_related_pages(text=text, current_page=page_path)

        page_body = [
            f"# {self._title_from_slug(page_slug)}",
            "",
            f"- category: {category}",
            f"- updated: {now}",
            f"- source: {source_rel}",
            "",
            "## Summary",
            summary,
            "",
            "## Evidence",
            *[f"- {line}" for line in evidence],
            "",
            "## Source",
            f"- [{source_rel}]({source_rel})",
            "",
            "## Related Pages",
            *self._format_related_lines(related_pages),
            "",
            "## Notes",
            "- Keep this page aligned with future ingest updates and linked concepts.",
            "",
        ]
        page_path.write_text("\n".join(page_body), encoding="utf-8")
        self._apply_backlinks(new_page=page_path, related_pages=related_pages)

        self._upsert_index_entry(category, page_path, summary)
        self._append_log(f"ingest | {source_rel} | updated {self._to_repo_relative(page_path)}")

        return "\n".join(
            [
                "# Wiki Ingest",
                "",
                f"- source: {source_rel}",
                f"- category: {category}",
                f"- page: {self._to_repo_relative(page_path)}",
                "",
                "## Summary",
                summary,
            ]
        )

    def ingest_batch(self, folder_path: str) -> str:
        self.init()
        resolved = self._resolve_source_path(folder_path)
        if not resolved.exists() or not resolved.is_dir():
            return f"Wiki batch ingest failed: folder not found: {resolved}"

        supported_suffixes = {".md", ".txt", ".log", ".rst"}
        sources = [
            path
            for path in sorted(resolved.rglob("*"))
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
        if not sources:
            return f"Wiki batch ingest skipped: no supported files in {self._to_repo_relative(resolved) if resolved.exists() else resolved}"

        results: list[str] = []
        for source in sources:
            result = self.ingest(str(source))
            first_line = result.splitlines()[0] if result else "Wiki Ingest"
            results.append(f"- {self._to_repo_relative(source)} -> {first_line}")

        self._append_log(f"ingest-batch | {self._to_repo_relative(resolved)} | files={len(sources)}")
        return "\n".join(
            [
                "# Wiki Batch Ingest",
                "",
                f"- folder: {self._to_repo_relative(resolved)}",
                f"- files_ingested: {len(sources)}",
                "",
                "## Results",
                *results,
            ]
        )

    def query(self, question: str) -> str:
        self.init()
        cleaned_question = question.strip()
        if not cleaned_question:
            return "Wiki query failed: question is empty."

        results = self._search_pages(cleaned_question)
        if not results:
            self._append_log(f"query | {cleaned_question} | no-results")
            return "\n".join(
                [
                    "# Wiki Query",
                    "",
                    f"Question: {cleaned_question}",
                    "",
                    "No matching wiki pages were found. Ingest additional sources first.",
                ]
            )

        top_results = results[:3]
        lines = [
            "# Wiki Query",
            "",
            f"Question: {cleaned_question}",
            "",
            "## Answer",
        ]
        for item in top_results:
            rel = self._to_repo_relative(item.path)
            lines.append(f"- {item.snippet} (source: [{rel}]({rel}))")

        lines.extend(
            [
                "",
                "## Citations",
                *[f"- [{self._to_repo_relative(item.path)}]({self._to_repo_relative(item.path)})" for item in top_results],
            ]
        )

        self._append_log(f"query | {cleaned_question} | {len(top_results)} citations")
        return "\n".join(lines)

    def lint(self) -> str:
        self.init()
        pages = self._all_content_pages()
        inbound_counts = self._compute_inbound_links(pages)

        orphan_pages: list[str] = []
        thin_pages: list[str] = []
        missing_evidence: list[str] = []
        weak_crosslinks: list[str] = []
        contradictions = self._detect_experiment_contradictions(pages)

        for page in pages:
            rel = self._to_repo_relative(page)
            text = page.read_text(encoding="utf-8", errors="replace")
            char_count = len(text.strip())
            link_count = len(re.findall(r"\[[^\]]+\]\(([^)]+)\)", text))

            if inbound_counts.get(page, 0) == 0:
                orphan_pages.append(rel)
            if char_count < 280:
                thin_pages.append(rel)
            if ("/experiments/" in rel or "/failures/" in rel) and "## Evidence" not in text:
                missing_evidence.append(rel)
            if link_count < 1:
                weak_crosslinks.append(rel)

        findings = [
            "# Wiki Lint",
            "",
            f"- scanned_pages: {len(pages)}",
            f"- orphan_pages: {len(orphan_pages)}",
            f"- thin_pages: {len(thin_pages)}",
            f"- missing_evidence_sections: {len(missing_evidence)}",
            f"- weak_crosslinks: {len(weak_crosslinks)}",
            f"- contradictory_experiment_claims: {len(contradictions)}",
            "",
            "## Findings",
            f"- Orphans: {', '.join(orphan_pages) if orphan_pages else 'none'}",
            f"- Thin pages: {', '.join(thin_pages) if thin_pages else 'none'}",
            f"- Missing evidence: {', '.join(missing_evidence) if missing_evidence else 'none'}",
            f"- Weak cross-links: {', '.join(weak_crosslinks) if weak_crosslinks else 'none'}",
            f"- Contradictions: {'; '.join(contradictions) if contradictions else 'none'}",
            "",
            "## Recommended Fixes",
            "- Add links from index and related pages to every orphan page.",
            "- Expand thin pages with measured outcomes and explicit evidence.",
            "- Ensure experiment and failure pages include an Evidence section.",
            "- Add at least one related-page link to each wiki page.",
            "- Reconcile contradictory experiment claims by recording setup deltas and evidence provenance.",
        ]

        self._append_log("lint | completed health-check")
        return "\n".join(findings)

    def _resolve_source_path(self, source_path: str) -> Path:
        candidate = Path(source_path)
        if candidate.is_absolute():
            return candidate
        direct = (self.repo_root / candidate).resolve()
        if direct.exists():
            return direct
        return (self.wiki_root / "raw" / candidate).resolve()

    def _infer_category(self, text: str, source_path: Path) -> str:
        lowered = f"{source_path.name.lower()}\n{text.lower()}"
        checks = [
            ("failures", ("oom", "out of memory", "nan", "crash", "failure", "regression")),
            ("optimizers", ("optimizer", "adam", "adamw", "muon", "learning rate", "weight decay")),
            ("datasets", ("dataset", "tokenizer", "corpus", "tinystories", "data prep")),
            ("architectures", ("attention", "window", "depth", "architecture", "layer", "model")),
            ("hypotheses", ("hypothesis", "predict", "expect", "assume")),
            ("experiments", ("experiment", "val_bpb", "benchmark", "baseline", "run")),
        ]
        for category, keywords in checks:
            if any(keyword in lowered for keyword in keywords):
                return category
        return "experiments"

    def _build_page_slug(self, source_path: Path, text: str) -> str:
        stem = source_path.stem.strip().lower()
        if stem and stem not in {"log", "notes", "readme"}:
            return self._slugify(stem)
        tokens = self._tokenize(text)
        return self._slugify("-".join(tokens[:6]) or "wiki-entry")

    def _summarize_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "No textual content found in source."
        sentence_joined = " ".join(lines)
        sentences = re.split(r"(?<=[.!?])\s+", sentence_joined)
        clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if clean_sentences:
            return " ".join(clean_sentences[:3])
        return sentence_joined[:360]

    def _extract_evidence_lines(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        evidence: list[str] = []
        for line in lines:
            lowered = line.lower()
            if any(marker in lowered for marker in ("val_bpb", "loss", "oom", "improved", "regressed", "baseline", "tokens")):
                evidence.append(line)
            if len(evidence) >= 5:
                break
        if evidence:
            return evidence
        return lines[:3] if lines else ["No explicit evidence extracted from source."]

    def _upsert_index_entry(self, category: str, page_path: Path, summary: str) -> None:
        section_title = {
            "experiments": "## Experiments",
            "architectures": "## Architectures",
            "optimizers": "## Optimizers",
            "datasets": "## Datasets",
            "failures": "## Failures",
            "hypotheses": "## Hypotheses",
        }[category]

        rel = self._to_repo_relative(page_path)
        one_line = summary.replace("\n", " ").strip()
        if len(one_line) > 180:
            one_line = one_line[:177] + "..."
        entry_line = f"- [{page_path.stem}]({rel}) - {one_line}"

        content = self.index_path.read_text(encoding="utf-8", errors="replace")
        if entry_line in content:
            return

        lines = content.splitlines()
        out_lines: list[str] = []
        inserted = False

        for idx, line in enumerate(lines):
            out_lines.append(line)
            if line.strip() == section_title:
                j = idx + 1
                while j < len(lines) and not lines[j].startswith("## "):
                    out_lines.append(lines[j])
                    j += 1
                if entry_line not in out_lines:
                    out_lines.append(entry_line)
                inserted = True
                out_lines.extend(lines[j:])
                break

        if not inserted:
            out_lines.extend(["", section_title, entry_line])

        deduped: list[str] = []
        seen_entries: set[str] = set()
        for line in out_lines:
            if line.startswith("- ["):
                if line in seen_entries:
                    continue
                seen_entries.add(line)
            deduped.append(line)

        self.index_path.write_text("\n".join(deduped).strip() + "\n", encoding="utf-8")

    def _append_log(self, action: str) -> None:
        now = self._timestamp()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## [{now}] {action}\n")

    def _find_related_pages(self, text: str, current_page: Path, max_results: int = 3) -> list[Path]:
        tokens = set(self._tokenize(text))
        if not tokens:
            return []

        scores: list[tuple[int, Path]] = []
        for page in self._all_content_pages():
            if page.resolve() == current_page.resolve():
                continue
            page_text = page.read_text(encoding="utf-8", errors="replace")
            page_tokens = set(self._tokenize(page_text))
            overlap = len(tokens & page_tokens)
            if overlap <= 0:
                continue
            scores.append((overlap, page))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [page for _, page in scores[:max_results]]

    def _format_related_lines(self, related_pages: list[Path]) -> list[str]:
        if not related_pages:
            return ["- none"]
        lines: list[str] = []
        for page in related_pages:
            rel = self._to_repo_relative(page)
            lines.append(f"- [{page.stem}]({rel})")
        return lines

    def _apply_backlinks(self, new_page: Path, related_pages: list[Path]) -> None:
        if not related_pages:
            return
        new_rel = self._to_repo_relative(new_page)
        backlink_line = f"- [{new_page.stem}]({new_rel})"

        for page in related_pages:
            text = page.read_text(encoding="utf-8", errors="replace")
            if backlink_line in text:
                continue

            lines = text.splitlines()
            out: list[str] = []
            inserted = False
            for idx, line in enumerate(lines):
                out.append(line)
                if line.strip() == "## Related Pages":
                    j = idx + 1
                    while j < len(lines) and not lines[j].startswith("## "):
                        out.append(lines[j])
                        j += 1
                    out.append(backlink_line)
                    out.extend(lines[j:])
                    inserted = True
                    break

            if not inserted:
                if out and out[-1] != "":
                    out.append("")
                out.extend([
                    "## Related Pages",
                    backlink_line,
                ])

            page.write_text("\n".join(out).strip() + "\n", encoding="utf-8")

    def _detect_experiment_contradictions(self, pages: list[Path]) -> list[str]:
        claims_by_subject: dict[str, dict[str, list[dict[str, object]]]] = {}
        subject_stopwords = {
            "the", "a", "an", "and", "or", "for", "with", "from", "into", "that", "this", "was", "were",
            "run", "experiment", "baseline", "val_bpb", "improved", "regressed", "regression", "loss",
        }
        important_subject_tokens = {
            "adamw", "muon", "attention", "window", "sliding", "depth", "dataset", "tokenizer", "batch", "lr", "learning", "rate"
        }

        for page in pages:
            rel = self._to_repo_relative(page)
            if not any(segment in rel for segment in (
                "/experiments/",
                "/failures/",
                "/optimizers/",
                "/architectures/",
                "/datasets/",
            )):
                continue

            text = page.read_text(encoding="utf-8", errors="replace")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                lowered = line.lower()
                if "val_bpb" not in lowered:
                    continue

                direction = ""
                if any(token in lowered for token in ("improved", "better", "decreased", "reduced")):
                    direction = "improved"
                if any(token in lowered for token in ("regressed", "worse", "increased", "degraded")):
                    if direction == "improved":
                        direction = "mixed"
                    else:
                        direction = "regressed"
                if direction not in {"improved", "regressed"}:
                    continue

                tokens = [token for token in self._tokenize(lowered) if token not in subject_stopwords]
                subject_tokens = [
                    token
                    for token in tokens
                    if token in important_subject_tokens
                ]
                if not subject_tokens:
                    subject_tokens = tokens[:3]
                subject = " ".join(subject_tokens) if subject_tokens else "general"

                has_numeric_val_bpb = bool(re.search(r"val_bpb\s*(?:[:=]|from)?\s*\d", lowered))
                specificity = len(set(subject_tokens))

                claims_by_subject.setdefault(subject, {"improved": [], "regressed": []})
                claims_by_subject[subject][direction].append(
                    {
                        "source": rel,
                        "line": line,
                        "has_numeric_val_bpb": has_numeric_val_bpb,
                        "specificity": specificity,
                    }
                )

        contradictions: list[str] = []
        for subject, dirs in claims_by_subject.items():
            improved_claims = dirs.get("improved", [])
            regressed_claims = dirs.get("regressed", [])
            if improved_claims and regressed_claims:
                improved_sources = sorted({str(item["source"]) for item in improved_claims})
                regressed_sources = sorted({str(item["source"]) for item in regressed_claims})
                improved_list = ", ".join(improved_sources)
                regressed_list = ", ".join(regressed_sources)

                improved_has_numeric = any(bool(item.get("has_numeric_val_bpb")) for item in improved_claims)
                regressed_has_numeric = any(bool(item.get("has_numeric_val_bpb")) for item in regressed_claims)
                max_specificity = max(
                    [int(item.get("specificity", 0)) for item in improved_claims + regressed_claims] or [0]
                )

                if subject != "general" and improved_has_numeric and regressed_has_numeric:
                    severity = "high"
                elif subject != "general" and (improved_has_numeric or regressed_has_numeric or max_specificity >= 1):
                    severity = "medium"
                else:
                    severity = "low"

                contradictions.append(
                    f"severity={severity} subject='{subject}' improved_in=[{improved_list}] regressed_in=[{regressed_list}]"
                )

        contradictions.sort()
        return contradictions

    def _search_pages(self, question: str) -> list[WikiQueryResult]:
        tokens = set(self._tokenize(question))
        results: list[WikiQueryResult] = []

        for page in self._all_content_pages():
            text = page.read_text(encoding="utf-8", errors="replace")
            haystack = f"{page.name} {text}".lower()
            score = 0
            for token in tokens:
                if token in haystack:
                    score += haystack.count(token)
            if score <= 0:
                continue

            snippet = self._best_snippet(text, tokens)
            results.append(WikiQueryResult(path=page, score=score, snippet=snippet))

        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _all_content_pages(self) -> list[Path]:
        if not self.wiki_pages_root.exists():
            return []
        pages = [
            path
            for path in self.wiki_pages_root.rglob("*.md")
            if path.name not in {"index.md", "log.md"}
        ]
        pages.sort()
        return pages

    def _compute_inbound_links(self, pages: list[Path]) -> dict[Path, int]:
        inbound: dict[Path, int] = {page: 0 for page in pages}
        page_map = {self._to_repo_relative(page): page for page in pages}

        for page in pages:
            text = page.read_text(encoding="utf-8", errors="replace")
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
            for link in links:
                normalized = link.replace("\\", "/")
                target = page_map.get(normalized)
                if target:
                    inbound[target] = inbound.get(target, 0) + 1

        return inbound

    def _best_snippet(self, text: str, tokens: set[str]) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        if not lines:
            return "No textual snippet available."

        best_line = lines[0]
        best_score = -1
        for line in lines:
            lowered = line.lower()
            score = sum(1 for token in tokens if token in lowered)
            if score > best_score:
                best_score = score
                best_line = line
        return best_line

    def _to_repo_relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root.resolve()).as_posix()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug or "wiki-entry"

    def _title_from_slug(self, slug: str) -> str:
        return slug.replace("-", " ").title()
