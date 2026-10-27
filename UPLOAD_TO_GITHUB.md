# Copy/paste upload instructions (Mac)

The initial upload is PRIVATE while team/source/weight redistribution rights are
reviewed. Public visibility is a separate explicit step. Neither source nor
weights were uploaded by ChatGPT. Do not upload the large Docker archives.
This folder is new and must not be confused with earlier reconstruction folders.

## 1. Enter this folder

After unzipping the downloaded package in Downloads:

```bash
cd "$HOME/Downloads/VLM3D_CT_GitHub_Ready"
```

## 2. Install/sign into GitHub CLI

If gh is missing and Homebrew is installed:

```bash
brew install gh
```

Then:

```bash
gh auth login
gh auth setup-git
```

Choose GitHub.com, HTTPS and browser login. Do not paste tokens into chat.

## 3. Create and upload the source repository

```bash
git init -b main
git add README.md CHALLENGE_FORM.md UPLOAD_TO_GITHUB.md .gitignore tasks configs docs provenance scripts
git status --short
git commit -m "Document retained VLM3D CT submissions with source and configurations"
gh repo create vlm3d-ct-submissions --private --source=. --remote=origin --push
```

If Git asks for identity, configure name/email for THIS repository only, then
retry the commit. Use your actual identity and preferred GitHub commit email:

```bash
git config user.name "Peter Sauer"
git config user.email "REPLACE_WITH_YOUR_GITHUB_COMMIT_EMAIL"
```

Do not paste the placeholder email unchanged. If the remote repository name
already exists, stop and choose a different name; do not force-push over it.

## 4. Verify weights and prepare release assets

```bash
python3 scripts/github_release.py prepare \
  --repo "$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
```

This verifies all three original weight SHA-256s. Task1/3 remain in place.
Task2 is split into two new parts under Downloads, requiring about 2.9 GB extra
disk. The original archives remain unchanged. The local .release_plan.json is
ignored by Git and records only this upload session's local asset paths.

## 5. Upload the weight assets to a DRAFT release

```bash
python3 scripts/github_release.py upload
```

Keep the terminal open; several GB must transfer. No GPU is needed. The release
stays draft and the repository stays private. Rerunning resumes existing assets
when GitHub exposes their matching SHA-256; it never clobbers an existing asset.
If a partial upload fails, report the error instead of deleting source files.

## 6. Review and make public when ready

Open the repository:

```bash
gh repo view --web
```

Confirm team/affiliations, license/redistribution rights, no private data, and
remaining score mappings. No broad open-source license was chosen on your behalf.
Once these files are authorized for public sharing, the following commands make
the repository public and publish the weight release:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
gh release edit weights-v1 --draft=false
```

The challenge's public repository/weight-access requirements are not satisfied
by leaving a private repository and draft release inaccessible to reviewers.

## 7. Fill real public links into the form/docs and push them

```bash
python3 scripts/github_release.py finalize
git add README.md CHALLENGE_FORM.md docs/WEIGHT_DOWNLOADS.md provenance/ARTIFACTS.json
git commit -m "Link published submission weights and reassembly instructions"
git push
```

Finalize checks public visibility, published release state and remote asset
names/sizes; it checks server SHA-256 when the API supplies it. It does not
download and independently rehash the remote binary assets. It updates weight
availability to GitHub and writes the actual repository/download URLs. A local
unpublished plan never changes the form's availability claim.

## 8. Read the completed form responses

```bash
open CHALLENGE_FORM.md
gh repo view --json url --jq .url
gh release view weights-v1 --json url --jq .url
```

Use the repository URL in the challenge's GitHub field. Use each task's section
in docs/WEIGHT_DOWNLOADS.md for the weight link/filename field. Task2 needs both
parts, so its release page plus reassembly instructions is the appropriate link.
Before final form submission, confirm the two still-unresolved official score
mappings and any required training metadata; hosting does not resolve them.

## Sources for upload behavior

- https://cli.github.com/manual/gh_repo_create
- https://cli.github.com/manual/gh_release_create
- https://cli.github.com/manual/gh_release_upload
- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

GitHub documents each release asset must be under 2 GiB. These instructions
were checked on 2026-10-27. GitHub/service permissions and policies still apply.
