# Particle website

Live: [particle-local.vercel.app](https://particle-local.vercel.app).

Static marketing site matching the app’s ParticleDesign materials, PT Serif / Poppins / Space Mono, and #20241F dark appearance. Uses existing app PNG artwork and optimized WebP delivery. Fonts are self-hosted with their OFL notices. No analytics, cookies, remote fonts, API secrets or runtime dependencies.

## Build

With Node 22 or newer, run `node Website/build.mjs`. Serve `Website/dist` locally with any static server. The build renders the fourteen curated model cards from `models.json`. Website cards open their public model source; installation happens inside the Mac app. Chat/Code panels are labelled interface previews, not an online inference service.

## Vercel

Project: `particle-local` (`prj_bqJdaBV9qYnLMaQWrsvOAebW3XV3`), in Lakshya Prajapati’s team. Git source: MathObsession/Particle. Root directory: `Website`; build: `node build.mjs`; output: `dist`; install command: empty. Production alias is public; previews and deployment URLs retain Vercel authentication. `vercel.json` supplies security headers and asset caching. Deployment system environment supplies canonical/Open Graph URLs. No credentials are required.

The install/update command follows GitHub’s latest release, verifies the app archive through the published installer and preserves models and user data. Requirements and the current tester build’s notarization status are stated on the page.

## Checks

Desktop and mobile browser checks cover model filters/search, default identification, preview tabs, clipboard, FAQ, theme persistence, menu behavior, resource failures, JavaScript errors and horizontal overflow. Recheck these when changing interactions. Honor reduced motion and visible keyboard focus; no JavaScript is needed to read the static page or installation command.
