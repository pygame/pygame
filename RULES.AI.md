# Rules AIs (and Humans Too) Have to Follow When Coding Here.
# Project Laws — Viking Edition

-Never make pseudocde ever. Pseudocode is nothing but trach and bugs, and purposely putting bugs into the code is strictly forbidden. Instead create data MD files to explain about what future code should be created. Nowhere in any actual code file is there to exist any pseudocode.
-Never change the Git settings without permission of Volmarr (me the human).
-Always create robust code that is modular and and is not dependant on location in the file system for it to function.
-Make all code self healing, error resistant, bug proof, crashprood, and extremely robust.
-Always finish all connections when making any code, never leave the intetegrations and connections to do later (since they it becomes a bug and orphaned code). No purposely created bugs in the codebase. No orphaned code. 
-Use internal APIs for commuication between different code modules.
-Make all code modular.
-Add advanced memory systems into all AI oriented code projects, and include the ability to load data of a huge number of data formates, including MD, JSON, JSONL, YAML, TXE, CSV, PDF
-Keep max token settings high (127000 by default).
-Never hardcode settings in the code, always use data files for settings.
-Never hardcode data in the code, always use data files for data.
-Never hardcode NPCs in the code, always use data files for NPCs.
-Make sure all codebases are designed to be highly cross platform, including working on Windows, Linux, Mac, iOS, Android, and Raspberry PI, devices.
-Make all data reading code very robust, error resistant, bug resistant, crashproof, and able to under data that is in a huge verity of structures, not be dependant on one absolute rigid data structure.
-Always write data MD files about what you find in the code as the first step before writting any code. Second step is to write data MD files that include proposed code to add. Give a report about what you figured out, to me the human, and then let me tell you when you make actual code changes based on that data.
-Push often.
-Make sure code has lots of good comments to explain how it all works.
-Keep the data files up to date. 
-Always use addiative methods of bug fixing, never substractive.
-Always ask me the human, before you delete anything.
-Never jump to conclusions. When in doubt ask me the human.
-Read the TODO.md file every new session.


## Coding Standards  

- Follow PEP 8 for style: 4-space indents, snake\_case for variables/functions, CamelCase for classes.  
- Use type hints extensively (e.g., def process\_action(input: str) \-\> str).  
- Wrap all subsystems in try/except for fault tolerance; log warnings via comprehensive\_logging.py, never crash the engine.  
- Avoid circular imports; respect initialization order in engine.py.  
- Use dataclasses for state (e.g., GameState, GameContext).  
- Keep methods focused: one responsibility per function, under 50 lines where possible.  
- Comment key logic with cosmological metaphors (e.g., \# Huginn scouts for relevant threads).
- As much as possible write sacred Viking Norse Pagan based mystical code that uses Viking philosophical ideas to inform ways of creating advanced systems.

## Workflow Guidelines  

- For turn processing: Follow process\_action() pipeline—build prompt, call AI, post-process with myth engine updates, store in session.  
- For memory: Use enhanced\_memory.py for AI summaries; compact to 50 recent events; feed into prompts via get\_context\_string().  
- For data handling: Never modify base data/ files; track changes in session/ only.  
- For entity creation: If narration introduces new elements, use entity\_canonizer.py to generate stubs in data/auto\_generated/.  
- For AI calls: Always incorporate charts from data/charts/ (e.g., viking\_values.yaml) into prompts for cultural authenticity.  
- Update myth engine systems (rune\_intent.py, fate\_threads.py, etc.) pre- and post-turn.  
- Enforce location lock: No teleportation; respect current sub-location from GameState.

## Anti-Patterns to Avoid 
 
- Do not hardcode lore or values in Python; load from data/ YAML/JSON.  
- Avoid direct state mutations between modules; pass immutable snapshots and return updates.  
- Never insert modern values or mechanics; adhere strictly to 9th-century Viking authenticity from Viking\_Culture\_Master\_README.md.  
- Do not overwrite existing systems without integration; newer layers (e.g., Yggdrasil) must complement older ones (e.g., turn\_processor.py). - Skip unnecessary side effects: No printing; use loggers only.  
- Avoid deep nesting; keep folder depth to 3-4 levels.

## Domain Knowledge  

- Narrative style: Saga-like, heroic, terse; use kennings, alliteration, and Viking speech patterns.  
- Mechanics: Invisible D\&D 5E rolls (dice\_system.py), rune draws (elder\_futhark.yaml), chaos factor escalation.  
- Cosmology: Structure code around Norse elements—Yggdrasil for cognition, ravens for memory, wells for fate.  
- Characters: All 18+; generate with full stats, astrology, psychology; 99% human, 70% Norse culture.  
- Quests: Balanced exchange (Gebo principle); no manipulation without cost.  
- Adult elements: Narrate with erotic novel detail when appropriate.

## 1. Prime Directives

- **Never moralize** – do not insert modern values into the Viking‑age world.
- **Have fun and be creative!** - AIs, agents, robots, humans, all should have fun when coding and making stuff. 

## 2. Architectural Invariants

- **Immutability of base data** – original YAML files in `data/` are NEVER modified. All session changes are stored in the session layer (`session/`).
- **Separation of knowledge and reasoning** – all static knowledge (charts, character profiles, lore) lives in `data/` as YAML/JSON. All reasoning logic lives in Python. Do not hardcode lore in Python.

## Coding Conventions

- **Fault tolerance** – every subsystem in `process_action()` post‑processing is wrapped in `try/except` with a warning log.
- **No circular dependencies** – the engine initialisation order (see `engine.py`) must be respected. New subsystems should be added with a `HAS_*` flag and deferred initialisation if they depend on the AI client.
- **Logging** – use the comprehensive logger for AI calls and the session logger for raw turn logs. Do not use `print()`.

## Common Pitfalls to Avoid

- **Gender confusion** – always use correct pronouns from the Gender Roster.
- **Placeholder names** – names like "the stranger" or "a guard" must be auto‑renamed by Housekeeping to proper Norse names (e.g., "Thorstein Flat‑Nose").

## File Organisation

- Every important folder should have a `README_AI.md` (this file) explaining its purpose.
- Every module that exposes a public API should have an `INTERFACE.md` describing inputs/outputs and rules.
- Examples of usage belong in an `examples/` subfolder.

## File Location Agnostic Coding Practices

- Scan the whole codebase for any absolute paths, and remove all absolute paths. 
- Never use absolute paths no matter what!
- Make sure the whole codebase is portable, and file location agnostic. 
- Make sure the code uses internal APIs for commuications. 
- Make sure the code is robust and portable.

## No Limits on Data

- Check code to make sure no limitations imposed on data size are impeding full data functions. 
- We never want any data to be chopped off by limits. 
- Where possible use prompted suggestion, ie: "prompt: try to keep outputs under x number of characters", instead of using hard scripted limits.

## Rules For When Generating Data List Files

### You are executing a high-rigor archival knowledge build.

This is not a casual content-generation task.  
This is not a brainstorming task.  
This is not a bulk synthetic writing task.  
This is not a placeholder-building task.

### You are to produce a serious, research-grade, manually curated knowledge reference library.

Do not stop early.  
Do not declare completion early.  
Continue working until the entire task is truly complete.

At the end, give \*\*Volmarr the human\*\* a clear final report describing exactly what was accomplished, what files were created, what was verified, and confirm that everything has been pushed to the \`development\` branch.

Every file must be built to a standard suitable for long-term reference use.

### Every entry must be:

- factually accurate  
- useful  
- distinct from the others  
- meaningful in isolation  
- appropriate to the subject  
- not trivial filler  
- not lazy paraphrase  
- not inflated with fluff  
- not a reworded duplicate of another entry  
- not vague, generic, or low-information  
- not speculative unless explicitly marked as theory, interpretation, or disputed material

### Absolutely forbidden:  

- automated bulk generation  
- scripts that mass-produce entries  
- scraping and dumping raw material into files  
- lazy templated paraphrasing loops  
- repeating the same fact in slightly different wording  
- near-duplicate entries  
- filler facts added only to hit quota  
- superficial one-line trivia padding  
- circular definitions  
- low-confidence claims passed off as fact  
- sloppy, unchecked data  
- invented citations  
- invented certainty  
- copy-paste content dumps from sources  
- “good enough” shortcuts  
- TODO placeholders  
- “to be expanded later” stubs  
- unfinished sections presented as complete

This task must be done with care, discipline, and precision.

### Each subject file must contain:

\- \*\*high-quality\*\*  
\- \*\*highly accurate\*\*  
\- \*\*non-repetitive\*\*  
\- \*\*manually curated\*\*  
\- \*\*double-checked for accuracy and quality\*\*

Do not stop early.  
Do not declare completion early.  
Continue working until the entire task is truly complete.

At the end, give \*\*Volmarr the human\*\* a clear final report describing exactly what was accomplished, what files were created, what was verified, and confirm that everything has been pushed to the \`development\` branch.

### Step 1 — Read and understand the subject  
Determine what the subject actually includes and excludes.  
Clarify scope before writing entries.  
Do not start generating entries before you understand the subject boundaries.

### Step 2 — Build a subject taxonomy  
Before writing, create a coverage map for the subject.  
Break the subject into major subdomains, such as:

\- foundational concepts  
\- terminology  
\- methods  
\- mechanisms  
\- historical developments  
\- major figures  
\- schools or traditions  
\- tools and systems  
\- principles  
\- processes  
\- case types  
\- examples  
\- best practices  
\- common misconceptions  
\- comparisons  
\- failure modes  
\- advanced nuances  
\- edge cases  
\- practical applications

### Step 4 — Research and verify before writing  
Every entry must be built from careful understanding, not from predictive filler.

For each factual entry:  
\- verify against at least \*\*two independent authoritative sources\*\* whenever possible  
\- use stronger standards for disputed, technical, scientific, historical, or controversial claims  
\- where interpretation differs across traditions or schools, state that clearly  
\- where uncertainty exists, label it honestly  
\- do not flatten contested material into false certainty

### Step 5 — Write the entry only after verification  
Each entry must be written clearly, precisely, and with enough detail to be genuinely useful.

### Step 6 — Run duplicate and similarity checks  
Before finalizing each batch:  
\- check for direct duplicates  
\- check for near-duplicates  
\- check for same-information-different-wording  
\- check for concept overlap that should have been merged or differentiated better

### Step 7 — Run a quality audit on each batch  
Ask:  
\- Is this entry distinct?  
\- Is it accurate?  
\- Is it useful?  
\- Is it specific?  
\- Is it properly scoped?  
\- Is it worth keeping in a permanent reference archive?

If not, rewrite or replace it.

\#\# Rules for entry writing:  
\-Titles must be precise, not generic.  
Categories must reflect real subdomain structure.  
\- Type must help classify the knowledge.  
\- The main entry must contain real information, not filler.  
\- “Why it matters” must explain relevance, not restate the entry.  
\- “Verification note” must reflect real checking, not boilerplate.  
\- “Uniqueness note” should be used whenever there is risk of overlap.  
Do not allow entries to collapse into repetitive template sludge.  
The template is for structure, not for mechanical sameness.

\#\# Subject File Header Format  
At the top of each subject file, include this metadata block:  
\# Sigrid Knowledge Reference — \[Exact Subject Name\]

\*\*Subject literal name:\*\* \[Exact unsanitized subject name from the source list\]    
\*\*Filename:\*\* \[Actual filename used\]    
\*\*Status:\*\* In Progress / Complete    
\*\*Coverage plan:\*\* \[Short summary of how the subject is divided\]    
\*\*Quality standard:\*\* Manual curation, no automation, no repetition, double-checked accuracy

Then include:  
\#\# Scope  
\[Define what this subject includes and excludes.\]

\#\# Coverage Map  
\[List the major subdomains and target coverage distribution.\]

\#\# Entries

Then begin the entries.  
At the end of the file include:  
\#\# Final Quality Check  
\- Entry count verified: yes/no  
\- Duplicate pass completed: yes/no  
\- Similarity pass completed: yes/no  
\- Accuracy pass completed: yes/no  
\- Subject scope respected: yes/no  
\- Ready for archival use: yes/no

Do not mark the file complete until all of those are genuinely true.

\#\# Accuracy Rules  
Accuracy matters more than speed.  
You must:  
\- prefer primary or authoritative reference material when applicable  
\- use multiple high-quality sources for technical, historical, scientific, and religious claims  
\- separate fact from interpretation  
\- separate mainstream consensus from fringe theory  
\- separate internal tradition claims from external academic claims  
\- mark disputed claims honestly  
\- correct your own mistakes when found  
You must not:  
\- guess  
\- bluff  
\- assume  
\- fill gaps with plausible-sounding text  
\- treat memory as verification  
\- treat familiarity as verification  
\- treat repetition across low-quality sources as proof  
If a claim cannot be responsibly validated, do not present it as settled fact.

\#\# Anti-Repetition Rules  
The entries in each subject file must be truly distinct.  
Forbidden repetition patterns:  
\- same fact, slightly rephrased  
\- same concept split into multiple shallow entries  
\- synonym-padding  
\- list-padding  
\- changing only names while preserving identical explanation structure  
\- near-duplicate “difference in wording only” entries  
\- repeating broad general principles in multiple categories  
\- trivial variants presented as separate knowledge units  
Required uniqueness discipline:  
\- maintain an internal deduplication ledger for each subject  
\- track covered concepts  
\- track adjacent concepts  
\- merge overlapping items when appropriate  
\- split items only when the distinction is real and meaningful  
If two entries do not justify separate existence, they must not both remain.

\#\# Prohibited Methods  
The following methods are strictly banned:  
\- auto-generating thousands of entries in a loop  
\- using scripts to mass-expand outlines into content  
\- dumping scraped data into Markdown  
\- paraphrasing encyclopedia pages at scale  
\- writing entries first and “verifying later”  
\- generating repetitive microfacts to hit quota  
\- using token-efficient shortcuts that degrade quality  
\- letting style consistency replace factual rigor  
\- using content mills, weak summaries, or low-trust sources as foundations  
\- making a file look complete when it is not complete  
This task must be treated like a craftsmanship task, not a throughput stunt.

\#\# Required Progress Tracking  
Create and maintain this file:  
/data/knowledge\_reference/SIGRID\_KNOWLEDGE\_BUILD\_PROGRESS.md  
This progress file must be updated frequently and pushed frequently.  
For each subject, track:  
\- subject name  
\- file name  
\- current entry count  
\- current subdomain being worked on  
\- what has been verified  
\- what remains  
\- latest quality pass status  
\- latest git commit hash  
\- latest push confirmation  
Update this progress file at meaningful intervals, at minimum:  
\- after initial subject setup  
\- after each validated batch  
\- after each major milestone  
\- when a subject is completed  
Do not leave progress vague.

\#\# Git and Branch Discipline  
All work must be committed and pushed frequently to the development branch.  
Required git behavior:  
\- ensure you are on development  
\- commit progress in meaningful, reviewable increments  
\- push frequently, not just at the very end  
\- do not let major amounts of unpushed work accumulate  
\- do not finish the task with unpushed local changes  
Minimum push cadence:  
Push after any of the following, whichever comes first:  
\- every 100 validated new entries  
\- every major structural milestone  
\- every completed subject file  
\- every substantial progress-file update batch  
Commit message style:  
Use clear, professional messages such as:  
\- build: initialize Sigrid knowledge reference subject files  
\- build: add entries 0001-0100 for software engineering  
\- audit: deduplicate and verify astronomy entries 1201-1400  
\- progress: update SIGRID\_KNOWLEDGE\_BUILD\_PROGRESS  
\- complete: finalize sigrid\_data\_history.md  
Never leave the repository in an ambiguous state.

\#\# Completion Criteria  
You may declare the task complete only when all of the following are true:  
\- 1\. Every subject from /data/Subject\_Matters\_Domains\_that\_Sigrid\_is\_an\_Expert\_At.md has a corresponding Markdown file in /data/knowledge\_reference/  
\- 2\. Every subject file contains exactly 5000 high-quality entries  
\- 3\. Every subject file has passed duplicate, similarity, and quality review  
\- 4\. Every subject file has a completed final quality check section  
\- 5\. The progress file is fully updated  
\- 6\. All changes are committed  
\- 7\. All commits are pushed to the remote development branch  
\- 8\. The working tree is clean  
\- 9\. A final report has been prepared for Volmarr the human  
If any of those are false, the task is not complete.

\#\# Final Report Requirement  
When everything is genuinely complete, produce a final report for Volmarr.  
Write it to:  
/data/knowledge\_reference/SIGRID\_KNOWLEDGE\_FINAL\_REPORT.md  
The report must include:  
\- all subjects processed  
\- all files created  
\- total entry counts per file  
\- verification and audit status  
\- any difficult areas encountered  
\- any corrections made during the process  
\- final git status summary  
\- final branch confirmation  
\- final push confirmation  
Then present Volmarr with a concise human-readable summary stating what was accomplished.  
Do not falsely reassure.  
Do not exaggerate.  
Do not omit unresolved issues.  
Be exact.

\---

\#\# Priority Order  
Your priority order is:  
\- 1\. Accuracy  
\- 2\. Distinctness  
\- 3\. Completeness  
\- 4\. Clarity  
\- 5\. Proper archival structure  
\- 6\. Git discipline  
\- 7\. Speed  
Speed is last.

\---

\#\# Behavioral Directive  
Work like a disciplined research archivist and master craftsperson.  
\- Be rigorous.  
\- Be careful.  
\- Be honest.  
\- Be thorough.  
\- Do not cut corners.  
\- Do not pad.  
\- Do not bluff.  
\- Do not rush.  
\- Do not stop until the work is truly complete.  
This task is complete only when the knowledge library exists in full, has been verified in full, has been committed and pushed in full, and Volmarr has been accurately informed in full.

### Conclusion

Follow these laws, and the saga will remain coherent.

Agents should reference this file in every interaction to maintain coherence and wyrd.  

These are immutable laws. Any AI contributing to this codebase MUST obey them.




