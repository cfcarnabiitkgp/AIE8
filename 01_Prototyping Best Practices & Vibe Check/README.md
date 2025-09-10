<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Introduction and Vibe Check</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE8/tree/main/00_AIM_Quicklinks)

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|

## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** – these will be questions that you will be expected to gather the answer to! These can appear as general questions, or questions meant to spark a discussion in your breakout rooms!

- 🏗️ **Activities** – these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** – Take on a challenge! These builds require you to create something with minimal guidance outside of the documentation. Completing an Advanced Build earns full credit in place of doing the base assignment notebook questions/activities.

### Main Assignment

In the following assignment, you are required to take the app that you created for the AIE8 challenge (from [this repository](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge)) and conduct what is known, colloquially, as a "vibe check" on the application.

You will be required to submit a link to your GitHub, as well as screenshots of the completed "vibe checks" through the provided Google Form!

> NOTE: This will require you to make updates to your personal class repository, instructions on that process can be found [here](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)!


#### 🏗️ Activity #1:

<span style="color:red">Arnab: My AI engineering challenge app</span> [My Recipe Generator](https://arnab-ai-challenge-problem.vercel.app/)
<span style="color:red">
will need substantially different vibe check questions than the ones provided here (which are more for chatbot type apps). I will use N/A for the provided questions and instead list out 5 different questions that vibe checks different aspects of my app.
</span>

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner.
    - Aspect Tested: <span style="color:red"> N/A </span>
2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested:<span style="color:red"> N/A </span>
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested:<span style="color:red"> N/A </span>
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested:<span style="color:red"> N/A </span>
5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested:<span style="color:red"> N/A </span>

<span style="color:red"> List of my vibe-check questions: </span>

6. Generate a recipe using only "chicken, salt, pepper, oil" as ingredients for 4 people in 15 minutes with no dietary restrictions.

    - Aspect Tested: *Ingredient limitation handling and creative recipe generation with minimal ingredients.* In snapshot 1, you will see that only these basic ingredients are used generated recipe.

7. Generate a recipe using "rice, vegetables, soy sauce" for 2 people in 30 minutes, Chinese cuisine, vegetarian.

    - Aspect Tested: *Time constraint compliance and realistic cooking time estimation.* In snapshot 2, you will see that the total cooking time is constrained to be within 30 mins as requested by user.

8. Generate a recipe using "beef, brie, bread" for 6 people in 45 minutes, American cuisine, with "Vegan" and "Gluten-Free" dietary restrictions.

    - Aspect Tested: *Dietary restriction adherence and ingredient substitution logic.* Here user is asking for vegan and gluten free replacements that are similar to beef, brie cheese and bread. The LLM correctly comes up with options (see snapshot 3) that adheres to the dietary restrictions (using seitan strips with beef texture, vegan brie cheese that is chashew based, and chicpea crusted bread).

9. Generate a recipe using "tomatoes, basil, mozzarella" for 4 people in 30 minutes, Italian cuisine, vegetarian.

    - Aspect Tested: *Cuisine-specific recipe generation and cultural authenticity.* In snapshot 4, you will see that recipe and cooking steps  maintain authenticity to Italian cuisine.

10. Generate a recipe using "salmon, quinoa, spinach, feta cheese, lemon, garlic, olive oil, dill" for 2 people in 60 minutes, Mediterranean cuisine, no dietary restrictions.

    - Aspect Tested: *Complex ingredient integration and balanced recipe composition.* In snapshot 5, you will use a balanced (and complex) recipe is generated that utilizes all of the ingredients while satisifying time and cultural requirements.

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

#### A Note on Vibe Checking

>"Vibe checking" is an informal term for cursory unstructured and non-comprehensive evaluation of LLM-powered systems. The idea is to loosely evaluate our system to cover significant and crucial functions where failure would be immediately noticeable and severe.
>
>In essence, it's a first look to ensure your system isn't experiencing catastrophic failure.

#### ❓Question #1:

What are some limitations of vibe checking as an evaluation tool?
##### ✅ Answer: Vibe checking has several limitations as an evaluation tool:

1. Limited Scope: It only tests a small subset of possible inputs and scenarios, potentially missing edge cases or failure modes that could occur in production.

2. Subjective Assessment: The evaluation relies heavily on human judgment, which can be inconsistent and biased based on the evaluator's expectations and experience.

3. No Quantitative Metrics: It doesn't provide measurable performance indicators or allow for systematic comparison across different versions or models.

4. Surface-Level Testing: It focuses on obvious failures but may miss subtle issues like inconsistent formatting, minor logical errors, or gradual quality degradation.

5. No Stress Testing: It doesn't test system behavior under high load, concurrent users, or with malformed inputs.

6. Temporal Limitations: It's a snapshot in time and doesn't capture how the system performs over extended periods or with different data distributions.

7. Limited Reproducibility: Results may vary between different evaluators or testing sessions, making it difficult to track improvements consistently.

### 🚧 Advanced Build (OPTIONAL):

Please make adjustments to your application that you believe will improve the vibe check you completed above, then deploy the changes to your Vercel domain [(see these instructions from your Challenge project)](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge/blob/main/README.md) and redo the above vibe check.

> NOTE: You may reach for improving the model, changing the prompt, or any other method.

#### 🏗️ Activity #1
##### Adjustments Made:
- _describe adjustment(s) here_

##### Results:
1. _Comment here how the change(s) impacted the vibe check of your system_
2.
3.
4.
5.


## Submitting Your Homework
### Main Assignment (Activity #1 only)
Follow these steps to prepare and submit your homework:
1. Pull the latest updates from upstream into the main branch of your AIE8 repo:
    - For your initial repo setup see [00_Setting Up Git/README.md](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)
    - To get the latest updates from AI Makerspace into your own AIE8 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `01_Prototyping Best Practices & Vibe Check` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s01-assignment`
4. Edit this `README.md` file (the one in your `AIE8/01_Prototyping Best Practices & Vibe Check` folder)
5. Perform a "Vibe check" evaluation your AI-Engineering-Challenge system using the five questions provided above
6. For each Activity question:
    - Define the “Aspect Tested”
    - Comment on how your system performed on it.
7. Provide an answer to `❓Question #1:` after the `✅ Answer:` prompt
8. Add, commit and push your modified `README.md` to your origin repository.

>(NOTE: You should not merge the new document into origin's main branch. This will spare you from update challenges for each future session.)

When submitting your homework, provide the GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.

### The Advanced Build:
1. Follow all of the steps (Steps 1 - 8) of the Main Assignment above
2. Document what you changed and the results you saw in the `Adjustments Made:` and `Results:` sections of the Advanced Build's Assignment #1
3. Add, commit and push your additional modifications to this `README.md` file to your origin repository.

When submitting your homework, provide the following on the form:
+ The GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.
+ The public Vercel URL to your updated Challenge project on your AIE8 repo.
