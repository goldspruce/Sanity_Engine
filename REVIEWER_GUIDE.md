# Reviewer guide (optional expert step)

You are the final check on an automated adversarial review of a hypothesis. The engine searched the textbooks for counter-evidence and proposed objections. It cannot tell whether a quote truly supports an objection. You can.

Time: about 1-2 minutes per card, 10-15 cards. Stop early if you run out of time; unanswered cards are reported as NOT REVIEWED, never as cleared.

## What you receive (shared Drive folder for this case)
- `triage.md`: the cards, in blind form (no verdicts shown)
- `responses.csv`: open it as a Google Sheet; fill in `response`, optionally `note`, and your name in `reviewer`
- Notebook with the same books as the engine (see below)
- Do NOT open `report.md` until you have answered every card. Seeing the verdict first biases judgment.

## Notebook setup (once per domain)
1. Open the Notebook (NotebookLM or a Claude/Gemini Project) the team shared with you. If you are building one, add the files from `notebook_pack/`. They contain exactly the text the engine searched, so quotes on the cards will be findable.
2. Keep file names unchanged. Each card shows the original file name and a search string.
3. Check the tool's current limits on source size and count. If a book exceeds them, rebuild the pack with a smaller `--max-words`.
4. The notebook is an AI over the books. Use it to find passages fast, but confirm anything that changes your answer by reading the passage itself. It can paraphrase loosely or connect things the book does not.

## Answering cards
Each card has a type. Answer from the evidence on the card and the books, not from the engine's likely opinion.

| Type | Question | Allowed responses |
|---|---|---|
| I (issue) | Is this a real flaw in the hypothesis, and how serious? | `serious`, `minor`, `not_flaw`, `unclear` |
| E (entailment) | Does the quoted passage actually support the claim? | `supports`, `partly`, `not`, `unclear` |
| R (recall) | A model recalled this limit but search did not find it. Is it real, and is it in the books? | `in_books` (say where in `note`), `real_not_in_books`, `not_real`, `unclear` |
| U (untested) | No passage tested this. Can you settle it from knowledge? | `settled` (give the answer in `note`), `open_gap`, `out_of_expertise` |

Rules:
- A card marked **WARNING: quote NOT found** means the quoted text could not be located in the corpus. Treat the quote as possibly invented; search the notebook before judging.
- `unclear` and `out_of_expertise` are good answers. Do not guess.
- If you disagree with an objection, say so (`not_flaw` or `not`) and add a one-line reason. Disagreements help improve the tool.
- Judge the passage's wording, not whether the conclusion sounds right.
- Use exact response words from the table; other text is ignored.

## Returning your work
Save the sheet in the case folder (or export `responses.csv`). The person running the engine merges it with `python run.py --merge <case_folder> --responses responses.csv`.

## Sharing on Google Drive (for the project owner)
- Suggested layout: `TSE/notebook_pack/` (books), `TSE/cases/<case-id>/` (triage.md, responses sheet).
- Share with named people only, not "anyone with the link": hypotheses are unpublished and the books are licensed.
- Reviewers: Editor on their case folder, Viewer on `notebook_pack/`.
- Confirm that your book licenses and your institution's rules allow uploading the books to the notebook tool and storing them in Drive.

## Confidentiality
Hypotheses are unpublished work. Do not paste cards or reports into public tools.
