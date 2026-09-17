# Diarization Improvement Roadmap

Ce document sert de feuille de route et de prompt reutilisable pour ameliorer progressivement la detection, la diarisation et l'identification locale des intervenants dans Scriberr.

Contraintes permanentes :

- rester open source et self-hosted ;
- ne jamais envoyer audio, transcripts, embeddings, voiceprints ou metadonnees confidentielles vers une API SaaS ;
- privilegier les traitements locaux compatibles GPU NVIDIA ;
- exploiter la RTX 5090 quand cela apporte un gain reel ;
- preserver les informations originales separement des donnees corrigees ou derivees ;
- eviter les fallbacks cloud silencieux ;
- ne pas logger de contenu confidentiel ni de secrets.

## 1. VAD Et Segmentation Pre-Diarisation

Statut : implementation Silero disponible, gain de performance a valider sur le corpus local avant activation par defaut.

Notes :

- `pyannote` reste le comportement par defaut et utilise la segmentation interne du backend ;
- `silero` active un pre-VAD neuronal local, compacte les regions parlees avant diarisation, puis remappe les timestamps vers l'audio original ;
- les poids Silero sont inclus dans le paquet Python epingle : aucun telechargement de modele n'est effectue pendant la diarisation ;
- aucun gain de qualite ou de vitesse n'est affirme sans benchmark sur des audios representatifs ;
- NVIDIA/NeMo VAD pourra etre evalue plus tard cote environnement NVIDIA, mais Sortformer reste separe pour limiter la complexite immediate ;
- validation a refaire sur l'environnement cible avec Go, uv et Python disponibles.

Prompt futur :

> Inspecte le pipeline audio actuel de Scriberr, puis ajoute une etape VAD/segmentation locale avant diarisation. Compare Silero VAD, PyAnnote segmentation et NVIDIA/NeMo VAD selon qualite, vitesse, licence, compatibilite CUDA et complexite d'integration. Implemente la solution la plus coherente avec l'architecture existante, sans envoyer de donnees a l'exterieur. Expose des parametres prudents dans les profils de transcription si utile. Ajoute des tests et des logs de diagnostic sans contenu confidentiel.

Objectifs :

- reduire les silences inutiles ;
- limiter les faux changements de locuteur ;
- ameliorer les interventions courtes ;
- accelerer la diarisation sur les longs audios ;
- eviter les detections parasites dues au bruit.

## 2. Normalisation Et Qualite Audio

Statut : implemente avec analyse locale systematique et corrections optionnelles, desactivees par defaut.

Notes :

- l'audio original reste intact ; le pipeline produit au besoin un WAV temporaire mono 16 kHz ;
- `ffmpeg astats` detecte niveau trop faible, saturation et canaux vides sans journaliser de contenu audio ;
- la normalisation EBU R128 (`loudnorm`, cible prudente -16 LUFS) est configurable par profil ;
- une reduction de bruit FFT legere est disponible explicitement, sans activation automatique ;
- la conversion et les traitements sont appliques a la meme entree derivee pour l'ASR et la diarisation.

Prompt futur :

> Analyse le preprocessing audio existant, puis ajoute une couche optionnelle de normalisation locale avant ASR et diarisation : loudness normalization, verification du sample rate, mono 16 kHz stable, detection de saturation, audio trop faible, canaux vides, et eventuellement reduction de bruit legere via outils open source locaux. Ne detruis jamais l'audio original. Stocke ou logue seulement des metriques techniques non confidentielles.

Objectifs :

- stabiliser les entrees des modeles ;
- ameliorer ASR et diarisation sur reunions reelles ;
- detecter les fichiers problematiques ;
- eviter une correction audio agressive par defaut.

## 3. Diagnostics CUDA Et GPU Reels

Prompt futur :

> Ajoute des diagnostics fiables pour distinguer GPU disponible sur l'hote, GPU visible dans le process, et GPU reellement utilise par chaque modele. Journalise backend, modele, device, version CUDA/PyTorch si disponible, VRAM indicative et fallback CPU explicite. Si l'utilisateur demande CUDA et que le modele tombe en CPU, echoue ou avertis clairement selon le contexte. Ne logue aucun contenu confidentiel.

Objectifs :

- exploiter correctement la RTX 5090 ;
- eviter les fallbacks CPU silencieux ;
- faciliter le debug CUDA/PyTorch ;
- documenter les contraintes Blackwell/CUDA.

## 4. Post-Processing Diarisation Avance

Prompt futur :

> Renforce le post-processing de diarisation dans Scriberr. Pars de la logique actuelle d'attribution mot-par-mot et de speaker continuity. Ajoute des heuristiques conservatrices pour fusionner les segments adjacents du meme locuteur, ignorer les micro-segments non alignes a des mots, detecter les alternances A/B/A/B improbables, et conserver les vraies reponses courtes comme "oui", "non", "ok", "yes", "no". Centralise les seuils, documente les heuristiques et ajoute des tests unitaires.

Objectifs :

- reduire les erreurs de clustering visibles ;
- conserver les interventions courtes reelles ;
- ameliorer la lisibilite du transcript ;
- rendre les seuils maintenables.

## 5. Benchmark Local Des Backends

Prompt futur :

> Cree un outil de benchmark local pour comparer les backends de diarisation disponibles dans Scriberr : PyAnnote Community-1, NVIDIA Sortformer et autres options open source pertinentes. Mesure temps de traitement, nombre de speakers detectes, segments tres courts, mots non attribues, utilisation GPU et format de sortie. Ne requiers aucun service cloud et ne publie aucune donnee. Prevois une sortie JSON exploitable pour analyses futures.

Objectifs :

- choisir objectivement le backend par cas d'usage ;
- mesurer les regressions ;
- guider les presets utilisateurs ;
- preparer une evaluation DER/WER plus complete.

## 6. Alignement Mot-Par-Mot Robuste

Prompt futur :

> Audite la fiabilite des timestamps mots produits par chaque backend ASR. Si un backend ne fournit pas de timestamps mots fiables, ajoute une strategie locale : alignement dedie, degradation explicite vers attribution segment-level, ou avertissement clair. Ne fais pas croire a une attribution mot-par-mot precise quand les donnees source ne le permettent pas. Ajoute des tests avec changements de locuteur intra-segment.

Objectifs :

- eviter les attributions speaker trompeuses ;
- mieux gerer les changements de locuteur dans une phrase ASR ;
- conserver la qualite karaoke/timestamps ;
- rendre les limitations visibles.

## 7. Speaker Embeddings Locaux

Prompt futur :

> Prepare puis implemente une architecture locale pour speaker embeddings. Les embeddings et voiceprints sont des donnees biometrices sensibles : ne pas les stocker dans le transcript JSON principal, ne pas les logger, ne pas les envoyer a l'exterieur. Propose une structure locale pour embeddings, speaker_id temporaire, confidence, matched_known_speaker et similarity_score. Compare les modeles open source compatibles local/GPU avant integration.

Objectifs :

- stabiliser les intervenants sur longs audios ;
- preparer l'identification de speakers connus ;
- conserver une separation nette transcript/biometrie ;
- permettre une base locale `KnownSpeaker` plus tard.

## 8. Base Locale KnownSpeaker

Prompt futur :

> Concois une base locale `KnownSpeaker` pour l'identification future des intervenants : id, display_name, reference embeddings, creation date, metadata, consentement/activation si pertinent. L'identification doit rester entierement locale. Ajoute les migrations, repositories et API minimales sans exposer les vecteurs dans les reponses standards. Preserve les corrections manuelles de noms existantes.

Objectifs :

- reconnaitre des intervenants recurrents ;
- separer nommage manuel et identification automatique ;
- proteger les donnees biometrices ;
- permettre suppression/export local.

## 9. Presets De Profils Reunions

Prompt futur :

> Ajoute des presets de profils de transcription orientes reunions : 2-4 personnes, 5-10 personnes, 10-15 personnes, audio bruite, interventions courtes, GPU rapide. Chaque preset doit configurer uniquement des parametres justifies, sans inventer arbitrairement un nombre de speakers sauf dans un mode explicitement exact. Mets a jour l'UI en restant coherent avec le design existant.

Objectifs :

- simplifier la configuration utilisateur ;
- reduire les mauvais reglages ;
- adapter diarisation et ASR au contexte ;
- garder les profils modifiables.

## 10. Logs Qualite Sans Contenu Confidentiel

Prompt futur :

> Ajoute des logs et metriques qualite pour ASR/diarisation sans jamais inclure de transcript, audio, embeddings, tokens ou secrets. Logue backend, modele, device, duree de traitement, nombre de speakers, nombre de segments, micro-segments, mots sans speaker, timeline exclusive utilisee ou non, et warnings de fallback. Ajoute une vue ou un export de diagnostics si coherent.

Objectifs :

- faciliter le debug ;
- evaluer la qualite sans fuite de donnees ;
- identifier les fichiers difficiles ;
- aider a comparer les backends.

## 11. Evaluation Automatique Locale

Prompt futur :

> Mets en place une evaluation automatique locale pour diarisation et transcription. Prevois un petit corpus prive non commite ou un format de fixtures locales, puis calcule DER, WER, taux de mots non attribues, changements de speakers intra-segment et temps de traitement. Les tests doivent pouvoir etre sautes proprement si les modeles ou donnees privees ne sont pas presents.

Objectifs :

- mesurer les ameliorations ;
- detecter les regressions ;
- separer tests unitaires rapides et evaluations lourdes ;
- rester compatible avec donnees confidentielles locales.

## 12. Multitrack Et Diarisation Hybride

Prompt futur :

> Analyse le pipeline multitrack existant. Propose une strategie hybride ou les pistes separees servent de source forte pour l'identite speaker, puis la diarisation corrige les chevauchements, fuites micro et segments ambigus. Preserve les offsets, gains, pistes originales et corrections manuelles. Ajoute des tests sur fusion temporelle et attribution speaker.

Objectifs :

- exploiter les reunions avec pistes separees ;
- reduire les erreurs de diarisation quand l'identite de piste est connue ;
- mieux gerer chevauchements et bleed audio ;
- preserver les donnees originales.

## 13. Interface De Correction Speaker

Prompt futur :

> Ameliore l'interface de correction des speakers. Preserve les corrections manuelles, rends les changements faciles sur un segment ou une plage, et stocke ces corrections separement des sorties brutes. Prepare leur reutilisation future pour KnownSpeaker et embeddings locaux. Ne modifie pas agressivement l'UX existante sans inspecter les composants actuels.

Objectifs :

- rendre la correction humaine rapide ;
- preserver les corrections ;
- preparer l'apprentissage local futur ;
- ameliorer la confiance dans les transcripts finaux.

## Ordre Recommande

1. Diagnostics CUDA/GPU reels.
2. VAD et qualite audio.
3. Post-processing diarisation avance.
4. Benchmark local des backends.
5. Alignement mot-par-mot robuste.
6. Speaker embeddings locaux.
7. KnownSpeaker local.
8. Presets reunions.
9. Evaluation automatique locale.
10. Multitrack hybride.
11. Interface de correction speaker.
