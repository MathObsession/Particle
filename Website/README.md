# Particle website

Live: [particle-local.vercel.app](https://particle-local.vercel.app).

A compact static site: product preview, model selection and installation. The design follows Sarvam's restrained hierarchy and spacing while keeping Particle's mark and green palette. Uses self-hosted Newsreader / DM Sans, a single raster wash, and real Mac app captures with example content. No tracking, remote fonts, runtime dependencies or online inference demo.

## Build

Node 22 or newer: `node Website/build.mjs`. Serve `Website/dist` using a static server. `models.json` supplies the thirteen alternative model rows; Sarvam 30B remains the visible Default. Alternatives and installation details use native disclosure elements. The install/update command follows GitHub's latest release.

`prefers-color-scheme` controls the entire site's appearance and preview image, including live system changes. Dark background is #20241F. There is no theme toggle, theme script, stored theme or animation framework. The only JavaScript handles the copy button; reading, navigation, disclosures and appearance work without it.

## Assets

Newsreader and DM Sans come from [Google Fonts](https://github.com/google/fonts/tree/main/ofl), under their bundled OFL licenses. Latin WOFF2 subsets retain the required site punctuation. Newsreader uses weight 400 / optical size 48; DM Sans uses weights 400–600 / optical size 14. Existing app artwork supplies `wash.webp` and `logo.png`.

App images are captured from the installed app using isolated temporary state and `--render-preview <path> --preview-chat --preview-compact-chat`; add `--preview-dark` for dark. No user chats or real inference are captured. Resize captures to 1600 pixels wide and encode WebP quality 82. Both images share 1600 × 1075 dimensions, avoiding layout shifts when the system theme changes.

## Deployment

Vercel project `particle-local` (`prj_bqJdaBV9qYnLMaQWrsvOAebW3XV3`), MathObsession/Particle Git source, root `Website`, build `node build.mjs`, output `dist`, empty install command. Production alias is public; deployment and preview URLs remain authenticated. `vercel.json` supplies security headers. Vercel's system environment supplies canonical/Open Graph URLs. No API keys required.

## Review

Desktop/mobile checks cover both system themes, live theme changes, obsolete stored preferences, preview image loading, disclosure lists, exact clipboard text, anchor targets, missing resources and horizontal overflow. The redesign removes the former theme controls, mock interactive app, feature-card grid, orbit art, slogan sections, giant footer banner, search/filter interface and unused assets. Mac app releases and user data are unchanged.

Research: [Sarvam homepage](https://www.sarvam.ai/), [Venngage design critique](https://venngage.com/blog/ai-slop-in-design/), [Joshua Snoddy on repetitive AI websites](https://www.joshuasnoddy.com/blog/why-ai-websites-look-the-same/). These are design references, not evidence that a visual pattern proves AI authorship.
