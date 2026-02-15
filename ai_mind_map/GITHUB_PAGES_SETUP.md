# GitHub Pages Setup for AI Learning Path Visualizations

## How to Enable GitHub Pages

1. **Push your changes to GitHub:**
   ```bash
   git add .
   git commit -m "Add AI learning path visualizations with GitHub URLs"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your repository: https://github.com/Plamen5kov/voice-cloning-pipeline
   - Click on **Settings** (top right)
   - Scroll down to **Pages** in the left sidebar
   - Under **Source**, select:
     - Branch: `main`
     - Folder: `/ (root)`
   - Click **Save**

3. **Wait a few minutes** for GitHub to build and deploy your site

4. **Access your visualizations:**
   - Main site: `https://plamen5kov.github.io/voice-cloning-pipeline/`
   - Complete AI Learning Path: `https://plamen5kov.github.io/voice-cloning-pipeline/ai_mind_map/ai_complete_learning_path.html`
   - Voice Cloning Pipeline: `https://plamen5kov.github.io/voice-cloning-pipeline/ai_mind_map/ai_learning_mindmap_graph.html`
   - Simple Mindmap: `https://plamen5kov.github.io/voice-cloning-pipeline/ai_mind_map/ai_learning_mindmap.html`

## What Changed

All resource links in the HTML files now point to GitHub URLs like:
- `https://github.com/Plamen5kov/voice-cloning-pipeline/blob/main/README.md`
- `https://github.com/Plamen5kov/voice-cloning-pipeline/blob/main/01_python_programming/README.md`

This means when you click on resource links in the visualizations, they'll open the files directly on GitHub instead of trying to access local files.

## Alternative: Use Relative Paths for Local Viewing

If you also want to keep a local version that works with local files, you can:
1. Keep these HTML files with GitHub URLs for GitHub Pages
2. Create separate versions with relative URLs for local use
3. Or use JavaScript to detect if running locally vs on GitHub Pages and switch URLs accordingly

## Sharing Your Visualizations

Once GitHub Pages is enabled, you can share these interactive visualizations with anyone:
- No installation required
- Works on any device with a browser
- All resource links work properly
