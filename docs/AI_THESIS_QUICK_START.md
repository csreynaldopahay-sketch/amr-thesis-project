# AI-Assisted Thesis Writing - Quick Start

This is a quick reference guide for using the two-phase AI-assisted thesis writing system. For the complete guide, see [AI_THESIS_WRITING_GUIDE.md](AI_THESIS_WRITING_GUIDE.md).

---

## The Two-Phase System

**Phase 1:** AI analyzes your project files and creates a knowledge base  
**Phase 2:** AI writes thesis chapters using academic standards

---

## Step 1: Prepare Your Files

Create a ZIP archive containing:
- ✅ All source code
- ✅ Data files (or samples)
- ✅ Documentation (README, docs/)
- ✅ Requirements/dependencies
- ✅ Visualizations/results

---

## Step 2: Upload and Use Phase 1 Prompt

**Upload your ZIP file**, then paste this prompt:

```
Role: You are an expert Academic Researcher and Technical Thesis Consultant. Your task is to act as my co-author for a formal thesis manuscript based on the project files contained in the uploaded ZIP archive.

### Part 1: Initial Analysis

Before we begin writing, analyze the contents of the attached ZIP archive. Please perform the following steps:

1. **Inventory Check:** List the key files, source code, data, and documentation found.
    
2. **Project Understanding:** Identify the core objective, the problem it solves, the technologies/methodology used, and the primary results/outputs.
    
3. **Knowledge Mapping:** Create an internal summary of the project's logic flow so you can reference specific code snippets or data points in future chapters.
    

**Wait for my confirmation once the analysis is complete. Do not start writing chapters yet.**

### Part 2: Thesis Writing Protocol

Once I provide a command to write a specific chapter (e.g., "Write Chapter 1"), you will follow these guidelines:

- **Academic Tone:** Use a formal, objective, and scholarly voice. Avoid contractions and informal language.
    
- **Data Integration:** Use the actual logic, variables, and findings found in the ZIP files. Do not hallucinate features that don't exist in the project.
    
- **Standard Structure:** Follow a traditional thesis format (Introduction, Literature Review, Methodology, Implementation, Results, Conclusion) unless I specify otherwise.
    
- **Technical Accuracy:** When discussing the implementation, refer to specific algorithms, frameworks, or data structures found in the code.
    
- **Citation Placeholders:** Use [Citation Needed] or [Reference: Source Name] where external academic support is typically required.
    

### How to Proceed:

1. Analyze the ZIP file now.
    
2. Provide a **brief summary** (300-500 words) of your understanding of the project.
    
3. Ask me: **"Analysis complete. Which chapter should we begin with?"**
```

---

## Step 3: Request Chapters

After the AI completes its analysis, request specific chapters:

```
"Write Chapter 1: Introduction"
"Write Chapter 3: Methodology"
"Write Chapter 5: Results and Discussion"
```

---

## Step 4: Review and Refine

Review the generated content and request revisions as needed:

```
"Revise Section 3.2 to include more detail about the clustering algorithm"
"Add a table summarizing the preprocessing steps"
"Ensure the Results chapter aligns with the methodology"
```

---

## Key Points to Remember

✅ **Always verify technical accuracy** against your actual code  
✅ **Replace [Citation Needed]** placeholders with proper references  
✅ **Check that data references** match your actual results  
✅ **Maintain consistency** in terminology across chapters  
✅ **The AI is an assistant**, not a replacement for your expertise

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `"Write Chapter [N]: [Title]"` | Generate a complete chapter |
| `"Revise Section [X.Y] to..."` | Request specific section revision |
| `"Add more detail about..."` | Request elaboration |
| `"Ensure consistency with Chapter [N]"` | Cross-reference check |
| `"Use only information from the files"` | Prevent hallucination |
| `"Rewrite in more formal tone"` | Adjust academic style |

---

## Troubleshooting

**Problem:** AI adds features not in your code  
**Solution:** Say "Use only information from the analyzed files"

**Problem:** Tone too informal  
**Solution:** Say "Rewrite in formal academic tone"

**Problem:** Missing technical details  
**Solution:** Say "Add specific algorithm/function details from the code"

**Problem:** Inconsistent terminology  
**Solution:** Say "Match terminology from Chapter [N]"

---

## AMR Project-Specific Context

When working on this thesis, remind the AI of:

- **Encoding:** S=0, I=1, R=2
- **MDR Definition:** Resistant to ≥3 antibiotic classes
- **Key Files:** `main.py`, `src/`, `docs/METHODOLOGY.md`
- **Phases:** 2-8 (preprocessing through documentation)
- **Algorithms:** Hierarchical clustering, Random Forest, Logistic Regression
- **Terminology:** "Pattern Discrimination" not "Prediction", "Structure Identification" not "Clustering"

---

## Next Steps

1. ✅ Prepare your ZIP archive
2. ✅ Upload to AI assistant
3. ✅ Use Phase 1 prompt
4. ✅ Wait for analysis summary
5. ✅ Request chapters one by one
6. ✅ Review and refine each chapter
7. ✅ Replace citation placeholders
8. ✅ Final consistency check

---

**For detailed instructions, examples, and best practices:**  
👉 See the complete [AI_THESIS_WRITING_GUIDE.md](AI_THESIS_WRITING_GUIDE.md)
