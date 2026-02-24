# Atsushi Yanagisawa - Personal Homepage

A bilingual (English / Japanese) academic homepage built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, deployed on GitHub Pages.

## Prerequisites

- [Hugo Extended](https://gohugo.io/installation/) (v0.146.0 or later)
- [Git](https://git-scm.com/)

### Install Hugo

**Windows (winget):**
```bash
winget install Hugo.Hugo.Extended
```

**Windows (Chocolatey):**
```bash
choco install hugo-extended
```

**macOS (Homebrew):**
```bash
brew install hugo
```

**Linux (Snap):**
```bash
snap install hugo
```

Verify installation:
```bash
hugo version
```

## Local Development

1. **Clone the repository:**
   ```bash
   git clone --recurse-submodules https://github.com/AtsushiYanaigsawa768/mysite.git
   cd mysite
   ```

2. **Start the development server:**
   ```bash
   hugo server -D
   ```

3. **Open in browser:**
   - English: http://localhost:1313/en/
   - Japanese: http://localhost:1313/ja/

   The root URL (`http://localhost:1313/`) automatically redirects to the English page.

## Deploy to GitHub Pages

### Step 1: Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `mysite` (or any name you prefer)
3. Set to **Public**
4. Push your code:
   ```bash
   git remote add origin https://github.com/<YOUR_USERNAME>/mysite.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** > **Pages**
3. Under **Source**, select **GitHub Actions**
4. The deployment will trigger automatically on the next push to `main`

### Step 3: Verify Deployment

After a successful workflow run, your site will be available at:
```
https://<YOUR_USERNAME>.github.io/mysite/
```

### Note on Base URL

The `baseURL` in `hugo.yaml` is set to `"/"`. The GitHub Actions workflow automatically sets the correct base URL at build time using `--baseURL`. You do not need to change `hugo.yaml` for deployment.

If you use a custom domain, update the `baseURL` accordingly:
```yaml
baseURL: "https://your-custom-domain.com/"
```

## Project Structure

```
mysite/
├── hugo.yaml              # Site configuration (bilingual settings, menus, profile)
├── content/
│   ├── en/                # English content
│   │   ├── _index.md      # Homepage
│   │   ├── publications.md
│   │   ├── experience.md
│   │   ├── projects.md
│   │   ├── awards.md
│   │   └── search.md
│   └── ja/                # Japanese content (mirrors en/)
│       ├── _index.md
│       ├── publications.md
│       ├── experience.md
│       ├── projects.md
│       ├── awards.md
│       └── search.md
├── static/
│   └── slides/            # PDF slides for download
├── assets/css/extended/
│   └── custom.css         # Custom styles
├── layouts/partials/
│   └── extend_head.html   # Custom head extensions
├── themes/PaperMod/       # Theme (git submodule)
└── .github/workflows/
    └── hugo.yml           # GitHub Actions deployment
```

## Adding Content

### Add a PDF Slide

1. Place your PDF file in `static/slides/`:
   ```
   static/slides/my-presentation.pdf
   ```
2. Link to it in any content page:
   ```markdown
   [Download Slides](/slides/my-presentation.pdf)
   ```

### Add a New Publication

Edit `content/en/publications.md` (and `content/ja/publications.md` for Japanese) and add a new entry:

```markdown
### Paper Title

**Author 1**, **Author 2**

<span class="pub-venue">Venue Name, Year</span>

<span class="pub-tag">Tag1</span> <span class="pub-tag">Tag2</span>

Brief description of the paper.

[Paper](https://link-to-paper) | [Code](https://github.com/repo)
```

### Add a Profile Photo (Optional)

To add a profile photo, place an image at `static/images/profile.jpg` and add the following to each language's `profileMode` in `hugo.yaml`:
```yaml
imageUrl: "images/profile.jpg"
imageWidth: 150
imageHeight: 150
```

### Customize Profile

Edit the `profileMode` section in `hugo.yaml` for each language:

```yaml
languages:
  en:
    params:
      profileMode:
        title: "Your Name"
        subtitle: "Your subtitle"
```

## Features

- Bilingual support (English / Japanese) with language switcher
- Profile mode homepage with social icons
- Dark / Light mode toggle
- Full-text search
- Responsive design
- GitHub Actions auto-deployment

## License

Content is the author's own. Theme ([PaperMod](https://github.com/adityatelange/hugo-PaperMod)) is MIT licensed.
