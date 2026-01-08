# GitHub Pages Setup Instructions

Your midterm site has been pushed to GitHub. To make it accessible via GitHub Pages:

## Option 1: Enable GitHub Pages via Repository Settings (Recommended)

1. Go to your repository: https://github.com/jayoum1/Carbon-DBH
2. Click on **Settings** (top right)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Deploy from a branch**: `main`
   - **Folder**: `/midterm_site`
5. Click **Save**

Your site will be available at:
**https://jayoum1.github.io/Carbon-DBH/**

(Note: It may take a few minutes for GitHub to build and deploy)

## Option 2: Use GitHub Actions Workflow

If you want automatic deployment on every push:

1. Go to repository **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**
3. Save the settings
4. Then add the workflow file `.github/workflows/deploy-pages.yml` back to the repo

## Verify Deployment

After enabling Pages:
- Check the **Actions** tab to see the deployment status
- Visit your Pages URL (usually `https://[username].github.io/[repo-name]/`)
- The site should be live within 1-5 minutes

## Troubleshooting

**Site not loading?**
- Check the **Actions** tab for deployment errors
- Ensure the folder path is correct (`/midterm_site`)
- Verify all files are committed and pushed

**Need to update the site?**
- Make changes to files in `midterm_site/`
- Commit and push to `main` branch
- GitHub Pages will automatically rebuild
