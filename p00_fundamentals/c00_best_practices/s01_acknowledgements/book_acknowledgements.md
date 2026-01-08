### Acknowledgements when using Open Source Code
When you use open source code in your project, there are some best practices to follow to ensure proper attribution
and referencing.

A minimal “Gold Standard” checklist to follow is:
- License is compatible
- Credit in README.md
- License text preserved
- File headers for copied code
- Modifications declared

Below the items are explained in more detail, and practical advice on how to achieve them is provided.

#### 1. First: Check the License (Non-negotiable)
Before anything else, confirm the repository’s license:
Look for LICENSE, COPYING, or license headers in files.
If no license is present, you technically have no right to reuse the code (even with attribution)

**Common cases:**
- MIT / BSD / Apache-2.0 → reuse allowed with attribution
- GPL / AGPL / LGPL → reuse allowed, but with reciprocity requirements
- Creative Commons → often requires attribution, sometimes restricts commercial use

#### 2. Always Attribute Clearly (Even If Not Required)
Even when the license doesn’t strictly require it, attribution is the best practice, i.e. include 
a Credits / Acknowledgements section:

```
## Acknowledgements

This project uses code from:
- **Project Name** by Author Name
  https://github.com/username/repo
  Licensed under the MIT License
```

This is the most visible and widely accepted form of credit.

#### 3. Preserve License Text
Most permissive licenses require this, and you must include their license:
- If you copied files → keep the original license header in those files.
- If you copied multiple files → include their LICENSE in your repo

or append it to your own LICENSE file.

#### 4. Add File-Level Attribution (Strongly Recommended)
If you copied or adapted specific files:

```
# Portions of this file are derived from:
# https://github.com/username/repo
# Original author: Jane Doe
# License: MIT
# Changes: Function X was modified to do Y
```

This is especially important if:
- You modified the code
- Files may be reused independently of your repo

#### 5. Declare Modifications
Many licenses (Apache-2.0, GPL) require you to state changes, i.e. when you modified the 
original version of some code to add X and remove Y.

This can live in:
- File headers (as shown above)
- README
- NOTICE file

#### 6. Use a NOTICE or THIRD_PARTY.md File (Best for Larger Projects)
For projects with multiple dependencies or reused snippets:

```
# Third-Party Notices
This project includes code from:

## Project Name
- Repository: https://github.com/username/repo
- Copyright (c) 2022 Jane Doe
- License: Apache 2.0
```

#### 7. If You’re Using It as a Dependency (Not Copying Code)
If you are:
- Importing it as a library
- Using it as a Git submodule
- Referencing it via package manager (i.e. conda, pip)

Then:
- Mention it in README
- Keep its license intact

No need to add file-level comments unless you copied code

#### 8. When in Doubt, Over-Credit
Too much attribution is never a problem. Too little is.
