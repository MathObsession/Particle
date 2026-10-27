# Particle

ParticleOne is the SwiftUI desktop interface for Particle's Sarvam runtime. It uses the supplied Particle One wordmark silhouette, finished PNG artwork whose outer colors blend gently into the page, and background-free green interface marks. The Dock icon retains its macOS tile. Headings are PT Serif, interface and response text use Poppins, and dark code cards use Space Mono. Fonts and their SIL Open Font License texts are bundled; no system font installation is required.

## Build and launch

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/build_particleone.sh
open "$HOME/Applications/Particle.app"
```

The installed Swift 6.4 Command Line Tools can build the executables with `PARTICLE_TOOLCHAIN=clt Scripts/particle build -c release --product ParticleOne`. Packaging needs the full Xcode icon compiler for `Design/ParticleOne.icon`. Regenerate that document’s full-canvas PNG with `Scripts/particleone/make_app_icon.py`. Build outputs and compiler caches remain on the SD card. The packaged app includes its CLI engine, Metal resources and fonts; it does not require the developer's Xcode installation to run. Existing models on an external card still require that card.

Choose **Load model** to start one warm Sarvam process. An existing ParticleCLI session must be closed first; model ownership is enforced by the existing runtime lease. The SwiftUI process does not load a second model. Return sends a message, Shift-Return inserts a newline (input-method composition is respected), Command-Return also sends, Command-N starts a conversation and Command-period stops generation. Code and responses have copy actions; generated code is displayed and never executed. Direct answers is the default: an empty `<think></think>` assistant block is evaluated during prefill, avoiding generation of a long reasoning trace. A zero-token reasoning budget closes any reopened block in the actual model/KV stream. The chat bar exposes Thinking On/Off; On selects Brief thinking (64 tokens). Changing it keeps loaded weights and completed history, resets KV, and applies to the next reply. The control is disabled while loading or generating. Settings also offers Brief thinking (64 tokens) and Full thinking (the original unrestricted behavior). This is not display-only filtering; hard reasoning tasks may benefit from enabling thinking. Stop preserves the partial transcript, but excludes that incomplete pair from subsequent model history. Restoring a saved conversation replays its completed history with the official template.

Settings include model location, a configurable 2–6 active-expert budget, 6–16 expert slots, context capacity, output allowance and system guidance. Six experts remains the desktop default. The approved 18 GiB mirror is reused only when its receipt exists; ordinary installations use the downloaded pack directly without automatically duplicating it into a second cache. Capacity exhaustion reports an error and preserves the full transcript.

Saved chats, settings, profiles and memories live in a private SQLite database at `~/Library/Application Support/ParticleOne/particleone.sqlite` (file mode 0600). A first launch migrates the previous `workspace.json` atomically and retains that original file as a migration backup. Unreadable data blocks saving rather than being replaced. Export produces Markdown. The workspace menu creates and switches independent workspaces in the current database; it can also create or open a separate workspace folder. Model weights are shared, not copied for each workspace. Diagnostic logs remain bounded and in memory. No analytics or cloud inference is included.

## Running without the SD card

This Mac now uses the installed app and a complete internal model, including tokenizer and embeddings. Both saved workspaces were migrated without changing their chats or memories. A real reply succeeded while the SD card was unmounted; Xcode and build caches remain on the card. See [standalone installation and verification](standalone/README.md). No new app build was required.

## First-launch model installation

On launch, ParticleOne checks the saved location, its managed model folder and the existing Particle SD-card pack. If no usable model is present, the installation screen opens and begins the pinned public `.particle` release download. It can be paused/resumed, redirected to a different disk, or replaced with an existing model selection. The destination needs about 22 GB available. The publisher supplies `model-catalog.json` only after the approved public upload and remote verification finish.

Source: [MathObsession/sarvam-30b-particle](https://huggingface.co/MathObsession/sarvam-30b-particle). This is a native lossless runtime-layout pack, not a new quantization. Original model: Sarvam AI; source Q4_K_M quantization: mtrajan. A pinned `particle-distribution.json` lists all file paths, lengths and SHA-256 values. The installer rejects unsafe paths, wrong totals, unpinned catalogs and incomplete releases. It streams one file at a time directly to `.part` files, validates HTTPS redirects, resumes via byte ranges, verifies each completed file and atomically publishes the folder after writing a local verification receipt. Partial installations remain in a `.installing` folder; a failed download is never selected as a model. Credentials, local install progress and locks are excluded from the hosted release.

The public repository requires **no Hugging Face username or token for installation**. Publishing uses a transient authenticated process and never embeds publisher credentials in app resources, commands or model files. Do not paste access tokens into project files.

## Homebrew cask

Packaging writes the app archive, exact SHA-256 and a local Git-backed cask tap under `dist/homebrew-particleone`. Use the commands recorded in `dist/particleone-release.json`:

```sh
brew tap --custom-remote particleone/local file:///Users/lakshyaprajapati/Repositories/Particle/dist/homebrew-particleone
brew install --cask particleone/local/particleone
```

The local archive is ad-hoc signed. This is not an Apple-notarized public release, and no public tap has been published. A distribution release needs a Developer ID signature, notarization, a hosted app archive and a published tap. `package_app.py --release-url ... --homepage ... --identity ...` supports the packaging inputs; it does not pretend to perform notarization. See [Homebrew's Cask Cookbook](https://docs.brew.sh/Cask-Cookbook). The cask preserves user data on ordinary uninstall; explicit `--zap` removes ParticleOne's own support directory, including a managed downloaded model, but not Particle's shared expert mirror.

## Validation and visual review

`Tests/ParticleOneTests` covers JSONL transport fragmentation/Unicode/multiline input, literal CLI command text, restored-history roles, interrupted transcript replay, inert code/reasoning parsing, private workspace persistence, native default selection, release path traversal, pinned catalog/manifest validation, checksums and cancellation. Required SwiftUI macro guards remain in use via `StoredState`.

Dark appearance uses exactly `#20241F` for the workspace background. The supplied ParticleDesign hero is displayed as a PNG with a baked smooth edge-color blend that preserves its green background; light appearance retains its paper color.

The sidebar collapses to a slim icon access rail with Control-Command-S or the control beside the smaller app name. Hover over the rail’s app icon to expand it; clicking works too. Search, chats, workspaces, memory and settings remain accessible from the rail. Suggestions and the two welcome-copy lines have been removed. Before the first message, a pill composer sits near the center below the PNG banner. Sending the first message animates it into the larger composer at the bottom; New conversation restores the centered layout. Reduced-motion settings suppress these animations. Usernames are generated from readable adjective/role pairs, persist locally and can be edited in Settings. Appearance supports System, Light and Dark. Button labels include their surrounding padding in their hit targets.

Message actions copy, save a personal note or delete a message. Deleting a user question also removes its answer; deleting an answer leaves its question visible but excludes that pair from future context. Changes reset/rebuild context with warm weights. Memory is scoped to each workspace: explicit Save to memory opens an editable note, and Settings optionally enables conservative extraction of explicit first-person name/location/interests/preferences. Automatic saving is off by default, never reads assistant output, and does not save detected credentials or medical/financial details. Saved notes can be reviewed, edited and forgotten. At most 100 notes are stored; up to 12 recent notes (300 characters each) are supplied as hidden system context. These are local notes, not an encrypted vault or a claim to understand every personal disclosure. Deleting a chat does not automatically delete separately saved notes; forgetting them in Memory removes them from future model context.

The main top bar is removed. Model and workspace selectors live at the sidebar bottom, with matching bottom icons in the collapsed rail. Each new completed AI response stores its output-token count, decode rate and context count; its footer shows these values. Older replies and interrupted turns without recorded counts show that counts are unavailable. No counts are guessed from text. Conversation rows expose an ellipsis menu with Delete conversation, also available in the Conversation app menu. User-message pencil actions open Edit and regenerate; saving starts a new continuation from before the edited question, preserves the original chat, and generates a new answer on the warm worker. The unsent composer draft is retained.

The custom system prompt has a dedicated Settings box and never appears as a user chat message. Updating it resets KV while preserving visible conversation history. Code cards include bounded lexical syntax colors for common Swift, Python, JS/TS, JSON, shell and other language-tagged fences; copying always uses the original string. Unknown languages remain plain, inert text.

The model picker and Settings expose counts 2–6; changes reload the model safely and retain completed history. Replies and thinking support full Markdown with PT Serif headings, Poppins emphasis and Space Mono code. The footer reads Made in India, and the banner fades into the page.

The app has a model-free development smoke mode using `Scripts/particleone/mock_engine.py`, isolated temporary app data and an explicit engine override. It checks real child-process streaming, cancellation, restore, reload and font registration. It is not a model performance test. `--render-preview <PNG>` captures only the app's own rendered welcome screen; `--preview-chat` renders the illustrative conversation with its dark code card. Those previews load no model and display no fabricated speed measurements. Artwork can be reproduced with `Scripts/particleone/render_assets.py`; the installed design skill is unchanged.

## In-app updates

The updater-enabled build uses one stable app at `~/Applications/ParticleOne.app`. Signed releases appear through the top-right update icon, with manual controls in Settings and the app menu. See [update setup and publishing](UPDATES.md) for the one-time bootstrap, local release command and public hosting requirements.

## Current release

**Particle 0.3.0** adds Churo Chat and the working Particle Code / Churo Code tool harness. Open `~/Applications/Particle.app`, select Code, open a project and load the model. If requested, enable coding context first; six experts is recommended for tool reliability. [Guide](code/README.md) · [Validation](validation-0.3.0/README.md). The model stays internal at `~/Library/Application Support/ParticleOne/Models/churo-chat.particle`; old app-support/module identifiers remain for compatibility.

### Earlier releases

ParticleOne **0.2.3** adds signed in-app updates and a top-right availability icon, with one stable installation. See [update validation](validation-0.2.3/README.md). The previous **0.2.2** release removed the top bar, moved selectors to the sidebar bottom, added saved per-reply metrics and message editing. See [0.2.2 validation](validation-0.2.2/README.md). The previous [0.2.1 validation](validation-0.2.1/README.md) records the centered composer and access rail. The previous [0.2.0 validation](validation-0.2.0/README.md) records no-thinking, local memory, workspaces and palette checks. The previous release’s checks remain recorded in [0.1.1 validation](validation-0.1.1/README.md). The public model remains at revision `40febdc7327c6aa572af4cc01c9829c0ae68b2e1`. The app remains a local ad-hoc-signed build.

The older 0.2.1 build 2 in `dist/ParticleOne.app` uses a native Icon Composer asset to fill the macOS icon tile. The inactive 0.2.0 app/archive were removed. See [icon verification](icon-update/README.md).
