# Midterm Presentation Website

A polished single-page website for presenting the Pomfret Forest Growth & Carbon Simulation project.

## Quick Start

1. **Open the website locally:**
   ```bash
   # Navigate to the midterm_site folder
   cd midterm_site
   
   # Open index.html in your browser
   # On macOS:
   open index.html
   
   # Or simply double-click index.html in Finder
   ```

2. **Or use a local server (recommended for testing):**
   ```bash
   # Python 3
   python3 -m http.server 8000
   
   # Then open http://localhost:8000 in your browser
   ```

## Adding Plot Images

Place your R-generated plot images in the `figures/` folder with these exact filenames:

- `total_carbon_vs_years.png`
- `mean_dbh_vs_years.png`
- `dbh_distribution_by_year.png`
- `carbon_by_plot_over_time.png`

**To copy plots from your R visualization folder:**
```bash
# From project root
cp reports/r_visuals/figures/total_carbon_vs_years.png midterm_site/figures/
cp reports/r_visuals/figures/mean_dbh_vs_years.png midterm_site/figures/
cp reports/r_visuals/figures/dbh_distribution_by_year.png midterm_site/figures/
cp reports/r_visuals/figures/carbon_by_plot_over_time.png midterm_site/figures/
```

If an image is missing, the page will show a placeholder message.

## Customizing Content

### Update Footer
Edit `index.html` and replace:
- `[Your Name]` with your actual name
- `[Date]` with the presentation date

### Update Performance Metrics
If you have updated model metrics, edit the values in the Model section:
```html
<span class="metric-value">≈ 0.966</span>  <!-- R² -->
<span class="metric-value">≈ 3.80 cm</span> <!-- RMSE -->
<span class="metric-value">≈ 1.82 cm</span> <!-- MAE -->
```

## File Structure

```
midterm_site/
├── index.html          # Main HTML file
├── styles.css          # Stylesheet
├── README.md           # This file
├── assets/             # For additional assets (if needed)
└── figures/            # Place your plot images here
    ├── total_carbon_vs_years.png
    ├── mean_dbh_vs_years.png
    ├── dbh_distribution_by_year.png
    └── carbon_by_plot_over_time.png
```

## Features

- ✅ Responsive design (works on desktop, tablet, mobile)
- ✅ Smooth scrolling navigation
- ✅ Print-friendly layout
- ✅ Projector-optimized (large text, high contrast)
- ✅ Graceful image placeholders
- ✅ No external dependencies (pure HTML/CSS/JS)

## Browser Compatibility

Tested and works on:
- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers

## Tips for Presentation

1. **Fullscreen mode:** Press F11 (Windows/Linux) or Cmd+Ctrl+F (Mac) for distraction-free viewing
2. **Navigation:** Use the top navigation bar to jump to sections
3. **Printing:** The page is print-optimized if you need handouts
4. **Projector:** The design uses high contrast and large fonts for visibility

## Troubleshooting

**Images not showing?**
- Check that image files are in `midterm_site/figures/` folder
- Verify filenames match exactly (case-sensitive)
- Try opening via a local server instead of `file://`

**Styles not loading?**
- Ensure `styles.css` is in the same folder as `index.html`
- Check browser console for errors (F12)

**Navigation not working?**
- Ensure JavaScript is enabled in your browser
- Check browser console for errors
