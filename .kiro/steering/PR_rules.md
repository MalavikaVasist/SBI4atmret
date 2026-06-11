---
inclusion: always
---
<!------------------------------------------------------------------------------------
   Add rules to this file or a short description and have Kiro refine them for you.
   
   Learn about inclusion modes: https://kiro.dev/docs/steering/#inclusion-modes
-------------------------------------------------------------------------------------> 

# Git Workflow Rules

## Branching
- Never make changes directly on main or master.
- Before starting any task, create a new feature branch.
- Branch names should follow:
  - feature/<description>
  - fix/<description>
  - chore/<description>

## Pull Requests
- When work is complete, create a pull request.
- Do not merge pull requests.
- Wait for human review and approval before merging.

## Commits
- Make logical, atomic commits.
- Write clear commit messages.

## Safety Rules
- If currently on main, create and switch to a feature branch before editing files.
- Never bypass branch protection rules.

