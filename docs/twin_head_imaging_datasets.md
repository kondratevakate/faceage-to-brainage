# Twin head-imaging and face datasets

Проверенный реестр близнецовых данных, в которых есть MRI/fMRI/CT или иная
визуальная репрезентация головы: 2D/3D-лицо, landmarks, цефалограммы,
внутриротовые сканы, fundus/OCT, iris, thermal и video. Аудит ссылок:
**2026-07-20**. Воспроизводимая стратегия и границы полноты описаны в
[search log](twin_head_imaging_search_log.md).

Это полный реестр найденного в текущем поиске, а не доказательство отсутствия
других закрытых больничных коллекций. Число людей, пар и изображений не
взаимозаменяемо; в таблицах сохранена единица, указанная источником.

## Как читать статусы

| Статус | Значение |
|---|---|
| `OPEN` | Есть рабочая страница прямого скачивания или открытый репозиторий. |
| `LICENSE` | Данные выдаются после подписания лицензии. |
| `DUA` | Управляемый доступ: заявка, concept paper или data-use agreement. |
| `COMMERCIAL` | Рабочая карточка товара/запрос образца или коммерческий сбор. |
| `PAPER` | Подтверждена коллекция и статья, но рабочий маршрут к raw data не найден. |
| `UNKNOWN FACE` | MRI есть, но сохранность полного лица/FOV в выдаваемой версии надо проверять у владельца. |
| `NO FACE` | Источник прямо сообщает о defacing/brain masking; релиз не подходит для MRI-face. |

## Что раньше не учитывали

| Канал | Почему он важен | Что проверять |
|---|---|---|
| fMRI -> анатомический T1 | Большинство research fMRI-протоколов добавляет T1 для регистрации, но это не автоматическая гарантия. | Наличие `anat/`, нижняя граница FOV, defacing и согласие на facial reconstruction. |
| MEG/EEG -> T1 и digitizer | Индивидуальная MRI нужна для head model; digitizer даёт форму головы/опорные точки. | Raw T1, Polhemus/head-shape points и одинаковые ID. |
| PET -> CT/MRI | PET часто имеет attenuation CT либо отдельную T1. | Доступна ли исходная голова, а не только PET volume/brain mask. |
| Speech/vocal-tract MRI | T1 может сохранять нос, челюсть, нёбо, язык, гортань и шею. | Inferior FOV; sagittal coverage; dental artefacts. |
| Dental/orthodontic | CBCT, цефалограммы, STL-модели и 3D palate дают костную/мягкотканную форму лица. | Radiation modality, landmarks, occlusion и возможность связать с фото. |
| Eye imaging | Fundus/OCT/iris — визуальные биомаркеры старения головы, часто в тех же twin registries. | Совпадение участников с MRI/фото. |
| Dermatology/cosmetology | VISIA, UV, cross-polarized и perceived-age photos дают surface-age signal. | Raw images, timestamps и возраст в день съёмки. |
| Longitudinal biometrics | Повторные фестивальные фото/video полезнее одиночного портрета для face-age. | Stable subject IDs, год визита, camera/session metadata. |
| Registry linkage | Разные проекты одного twin registry могут оказаться пересекающимися. | Запрашивать overlap matrix по participant IDs, не предполагать совпадение. |

Главный вывод про fMRI: functional EPI сама по себе слишком грубая для лица. T1
действительно часто снимается в тот же визит, но публичный fMRI-релиз нередко
деидентифицирует именно T1. Поэтому сначала ищем не
`fMRI=yes`, а **original non-defaced T1 + полный anterior/inferior FOV + разрешение
на facial reconstruction**. Политика OpenNeuro, например, требует удаления
идентифицирующих facial features без специального согласия
([policy paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8550750/),
[user guide](https://docs.openneuro.org/user_guide.html)).

## P0/P1 shortlist

| Приоритет | Ресурс | Почему это лучший следующий запрос |
|---|---|---|
| P0 | RATSS | В одной шведской twin cohort подтверждены профессиональные медицинские фото лица и raw multimodal MRI; это самый прямой новый кандидат на paired face-brain analysis. |
| P0 | EMIF-AD PreclinAD / NTR | У 67 MZ-пар уже опубликован same-person анализ digital fundus images и brain MRI; broader cohort также имеет OCT, fundus autofluorescence, PET, MEG и rs-fMRI. |
| P0 | Monash T2D-discordant twins | 35 человек из 18 twin pairs имеют OCT, retinal photographs и 3T T1/FLAIR/DTI/SWI/ASL; небольшой, но полностью подтверждённый eye-brain target. |
| P0 | QTIM original/raw | На MRI этой когорты уже извлекали 3D landmarks верхней части лица; нужно запросить оригинальный non-defaced T1, а не полагаться на публичный релиз. |
| P0 | TwinsUK linkage | Есть большая 3D-face twin collection и новая MRI-кампания на 2,500 участников; нужно проверить пересечение ID и наличие полного лица в T1. |
| P0 | Danish LSADT | 1,826 пожилых близнецов с фронтальными фото, perceived age и mortality: самый прямой twin face-age ресурс. |
| P0 | Forsyth/Moorrees Twin Study | Почти 500 twin pairs с ежегодными записями роста, facial photographs, cephalograms и 3D study casts; потенциально лучший продольный FaceAge-архив. |
| P0 | Dunedin (не близнецы) | В одной возрастной фазе собраны стандартизованные facial photos и brain MRI; это сильнейшая прямая проверка face-age <-> brain-age. |
| P1 | TwinsMX | Более 270 близнецов уже прошли structural MRI, rs/task-fMRI, DWI и MRS; лучший найденный MRI-ресурс Латинской Америки. |
| P1 | Osaka linkage | В одном registry существуют MRI/fMRI/PET/MEG, dentistry и отдельный VISIA-проект; совпадение участников надо запросить. |
| P1 | NTR vocal tract | 632 Dutch twins; из обычных T1 уже извлечены 146 skull/face/mandible/palate/vocal-tract measures. |
| P1 | Lithuanian Twin Centre 3dMD | 106 DNA-typed twin pairs с textured 3D surfaces и 21 landmarks; более широкий craniofacial archive охватывает 141 пару с 3D face и/или cephalograms. |
| P1 | Notre Dame 3D-TEC | 107 twin pairs, 428 лицензируемых 3D scans; рабочий институциональный путь доступа. |
| P1 | Qlean `ds-016` | Реально продаваемая rights-cleared японская коллекция; 12 участников отмечены как MZ twins. |

## 1. Близнецовые MRI, fMRI, PET и MEG

| Ресурс и регион | Размер, подтверждённый источником | Модальности | Лицо в доступной версии | Доступ | Вывод |
|---|---:|---|---|---|---|
| [RATSS](https://ki.se/en/kind/research-at-kind/ratss-a-twin-study-of-autism-and-adhd), Швеция | Published MRI sample: 372 twins age 8-36; facial-morphology data for 280; current project reports >400 participants | T1, T2, FLAIR, DTI, rs-fMRI; professional 2D medical facial photographs | **Да в source cohort**; exact photo-MRI overlap, dates and distributed FOV need confirmation | `DUA` / collaboration; [methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7772336/) | **Лучший прямой paired lead.** NDD-enriched cohort, поэтому внешнюю валидность и diagnostic covariates надо планировать явно. |
| [EMIF-AD PreclinAD / Netherlands Twin Register](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091034/), Нидерланды | Amsterdam cohort: 100 complete pairs at baseline, 99 MZ; published retinal-MRI analysis: 134 people / 67 MZ pairs age 60+ | T1, 3D FLAIR, SWI, DWI, ASL, rs-fMRI, amyloid PET, MEG; digital fundus, OCT and fundus autofluorescence | MRI face not required for eye-brain use; raw head FOV/defacing remains unknown | `DUA`; requests via [EMIF-AD catalogue/project](https://www.emif.eu/emif-ad-2/) | **Сильнейший twin eye-brain lead:** retinal vascular parameters and MRI WMH already linked in complete MZ pairs. Ask for source images rather than derived traits only. |
| [Monash/ATR T2D-discordant twin study](https://www.nature.com/articles/s41598-022-13662-8), Австралия | 35 people from 18 pairs; mean age 63.8 | 3T T1 at 0.6 mm slice thickness, FLAIR, DTI, SWI, ASL; bilateral OCT and retinal photographs | MRI face status unknown and not needed for retinal-brain analysis | `DUA`; data may be available from corresponding author | Small but unusually complete same-visit eye-brain cohort with cognition, diabetes, vascular and anthropometric covariates. |
| [QTIM / OpenNeuro ds004169](https://github.com/OpenNeuroDatasets/ds004169), Австралия | Public cohort: 1,202 twins/siblings; facial-landmark analysis: 545 twin individuals, heritability subset 79 MZ + 90 DZ pairs | T1, rs-fMRI; test-retest subset | `UNKNOWN FACE`: public OpenNeuro copy нельзя считать сохранённой без voxel audit | `OPEN` public; original/raw надо запрашивать | **Уникально:** в исходных 3D head MRI уже размечали 9 upper-face landmarks; нижнее лицо не входило в FOV ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3441666/), [release study](https://academic.oup.com/cercor/article/33/22/11060/7285338)). |
| [QTAB](https://pmc.ncbi.nlm.nih.gov/articles/PMC10082846/), Австралия | 422 baseline, 304 follow-up, age 9-16 | MP2RAGE T1, T2, FLAIR, SWI, DWI, ASL, rs/task-fMRI | `NO FACE`: статья прямо сообщает removal facial features из 3D T1/T2/FLAIR | `OPEN` ([repository mirror](https://github.com/OpenNeuroDatasets/ds004146)) | Хороший twin brain-development benchmark, не MRI-face. |
| [CATS](https://www.nature.com/articles/s41597-025-05449-z), Китай | 136 adolescent twins; 34 MZ + 30 DZ confirmed complete pairs | sMRI, rs-fMRI, dMRI | `NO FACE`: facial features obscured | `DUA` через Brain Science Data Center; [DUA record](https://zenodo.org/records/15335779) | Сильный азиатский twin brain dataset; лицо из релиза не восстановить. |
| [HCP Young Adult](https://humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release), США | 1,206 adults, twins and siblings from family design | T1/T2, rs/task-fMRI, dMRI; MEG/7T subsets | `NO FACE`: official release defaced and de-eared ([access guide](https://www.humanconnectome.org/study/hcp-young-adult/document/quick-reference-open-access-vs-restricted-data)) | `OPEN` imaging + restricted family metadata | Лучший масштабный twin connectomics benchmark, не источник лица. |
| [TwinsMX](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/1946988F44E1D0D17C21F61EFA42D29D/S1832427424000185a.pdf/twinsmx_exploring_the_genetic_and_environmental_influences_on_health_traits_in_the_mexican_population.pdf), Мексика | Registry 2,826 people, 450 complete pairs; >270 twins with MRI | High-res anatomy, 10-min rs-fMRI, working-memory fMRI, DWI, MRS | `UNKNOWN FACE` | `DUA` / collaboration | Главная новая находка для Латинской Америки; запросить raw T1 FOV и defacing policy. |
| [Osaka University Twin Registry](https://pubmed.ncbi.nlm.nih.gov/31910931/), Япония | Registry reports 259 MZ + 39 DZ pairs; modality-specific N varies | Brain MRI, fMRI, PET, MEG; dentistry, dermatology, ultrasound | `UNKNOWN FACE` | `DUA` / collaboration | Особенно ценен потенциальным cross-project linkage; совпадение VISIA/MRI не подтверждено. |
| [Swedish affective face-matching twin study](https://www.nature.com/articles/s41398-026-04240-x), Швеция | 270 twins: 64 MZ + 71 DZ pairs | Whole-head T1 + face-matching task fMRI | `UNKNOWN FACE` | Data on reasonable request | Новая 2026 когорта с явным whole-head T1; запросить raw non-defaced scan. |
| [TWIN-E / TWIN-10](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252211/), Австралия | >1,600 overall; 270 MRI participants | Structural, DWI, proton density, 4 fMRI tasks | `UNKNOWN FACE` | `DUA`; [NeuRA project](https://neura.edu.au/project/heritability-emotion-cognition-twins) | Мультимодально и глубоко фенотипировано; raw facial coverage не опубликован. |
| [VETSA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3780387/), США | 526 male twins with 3D structural MRI at wave 1 | T1; later resting/default-mode fMRI | `UNKNOWN FACE` | `DUA` / collaboration | Сильная aging/twin когорта, только мужчины. |
| [NTR vocal-tract mega-sample](https://pmc.ncbi.nlm.nih.gov/articles/PMC9672028/), Нидерланды | 632 twins: 290 complete pairs + 48 unpaired | Structural T1; 146 landmark/semi-landmark phenotypes | **Да в original:** skull, face, mandible, hard palate, nasal cavity, tongue/larynx | `DUA` через [NTR](https://ntrdata.nl/); [code](https://github.com/ddediu/vt-heritability) open | Лучшее доказательство, что routine T1 может содержать полезную нижнюю часть головы. |
| [BrainSCALE](https://academic.oup.com/cercor/article/29/3/978/4824648), Нидерланды | 112 twin families, longitudinal | Structural MRI | `UNKNOWN FACE` | Request corresponding author | Развитие мозга; ценность зависит от original FOV. |
| [Older Australian Twins Study MRI sample](https://researchers.mq.edu.au/en/publications/incidental-findings-on-cerebral-mri-in-twins-the-older-australian), Австралия | MRI analysis reported 320 people with co-twin: 92 MZ + 68 DZ pairs | Repeated structural brain MRI, cognition and health | `UNKNOWN FACE` | `DUA` / collaboration | Полезна для brain-age и longitudinal aging; original image access надо согласовывать с OATS investigators. |
| [Early Brain Development in Twins, NDA #1974](https://nda.nih.gov/edit_collection.html?id=1974), США | >100 pairs recruited; 248 subjects with shared imaging | Prenatal ultrasound, neonatal/longitudinal structural MRI, DTI | `UNKNOWN FACE`; infant FOV must be inspected | `DUA` via NDA | Реальный controlled dataset, не просто статья; важен для craniofacial development, но не adult age. |
| [NIMH pediatric twin MRI](https://www.nimh.nih.gov/news/science-updates/2014/genes-impact-suspect-cortex-areas-more-as-youth-mature), США | 792 twins and siblings; 1,748 MRI scans, up to 8 timepoints | Longitudinal structural MRI | `UNKNOWN FACE` | Cohort/request; exact current image route must be confirmed | Очень сильная longitudinal design; не путать с general NIH pediatric MRI database. |
| [PETS-11](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/336452A9F36F1DF9C2721325A32B3AA4/S1832427419001208a.pdf/epigenetic_influences_on_neurodevelopment_at_11_years_of_age_protocol_for_the_longitudinal_peripostnatal_epigenetic_twins_study_at_11_years_of_age_pets11.pdf), Австралия | 151 same-sex pairs offered optional MRI | MRI + cognition + epigenetics | `UNKNOWN FACE`; actual scanned N must be requested | `DUA` / collaboration | Lead, не готовый image release. |
| [ADTWIN / Older Finnish Twin Cohort](https://www.cambridge.org/core/journals/twin-research-and-human-genetics/article/older-finnish-twin-cohort-45-years-of-followup/94760871D9703AF3A974BB8E32B0180F), Финляндия | Older twins discordant for cognition; modality N in substudies | MRI + PiB/FDG PET + neuropsychology | `UNKNOWN FACE` | `DUA` via THL Biobank/collaboration | PET может добавить CT/MRI coregistration; нужно спросить, какие source volumes сохранены. |
| [HUMMUS / Swedish Twin Registry](https://www.cambridge.org/core/journals/twin-research-and-human-genetics/article/swedish-twin-registry-a-research-database-in-the-era-of-new-swedish-legislation/5C04CCF8B13CD1DA5977A7D003FCE120), Швеция | MRI subset linked to questionnaire and piano-training study | Brain MRI + behavioral phenotypes | `UNKNOWN FACE` | `DUA`; [public record](https://researchdata.se/en/catalogue/dataset/2025-175) is questionnaire-only | Не считать MRI открытой: доступная карточка не содержит изображений. |
| [Minnesota Twin Family Study](https://mctfr.psych.umn.edu/our-research/twin-family-study), США | 1,400 original + 500 additional twin pairs; MRI subset N not stated on access page | MRI, neurocognition, psychophysiology, genomics | `UNKNOWN FACE` | Collaboration | [Center overview](https://mctfr.psych.umn.edu/about-mctfr) confirms MRI, но image inventory надо запросить. |

## 2. Близнецовые лицо, 3D, landmarks и biometrics

| Ресурс и регион | Размер | Репрезентация головы | Доступ | Вывод |
|---|---:|---|---|---|
| [RATSS facial photographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC7772336/), Швеция | 280 twins with facial-morphology data within a 372-person MRI sample | Standardized 2D medical photographs transferred to Face2Gene; linked T1/T2/FLAIR/DTI in cohort | `DUA` / reasonable request through [RATSS](https://ki.se/en/kind/research-at-kind/ratss-a-twin-study-of-autism-and-adhd) | Запросить raw photographs rather than platform-derived dysmorphology scores, exact same-visit overlap and consent scope. |
| [Danish LSADT](https://pmc.ncbi.nlm.nih.gov/articles/PMC4881821/), Дания | 1,826 twins age 70-99 | Standardized frontal color photos, perceived age, mortality | `DUA`; [registry guidelines](https://www.sdu.dk/en/forskning/dtr/researcher/guidelines) | Самая прямая найденная twin face-age cohort. |
| [Forsyth/Moorrees Twin Study](https://www.aaoflegacycollection.org/aaof_collection.html?id=Forsyth), США | Historical archive: nearly 500 twin pairs; secondary inventory reports 533 families; annual records from childhood into adulthood | Facial photographs, lateral/frontal cephalograms, dental records and 3D study casts | Online AAOF materials plus permission/full-resolution request via [contact page](https://www.aaoflegacycollection.org/aaof_contacts.html) | **Сильный longitudinal FaceAge target.** До анализа проверить, сохраняют ли public subject IDs twin pairing/zygosity и какие modalities overlap по визитам. |
| [TwinsUK 3D face](https://pmc.ncbi.nlm.nih.gov/articles/PMC5008732/), Великобритания | 1,380 analyzed women; 263 MZ + 341 DZ complete pairs | 3dMD surface, 37 manual + derived landmarks; dense maps ~4,249 points | `DUA`; [resource page](https://twinsuk.ac.uk/researchers/explore-our-data-and-samples/external-repositories/) | Новая MRI-кампания планирует 2,500 scans ([announcement](https://www.kcl.ac.uk/news/twinsuk-first-mri-scan-study-healthy-ageing)); overlap пока не доказан. |
| [Notre Dame twin datasets](https://cvrl.nd.edu/projects/data/), США | 3D-TEC: 107 pairs/428 scans; ND-TWINS: 24,050 photos/435 attendees; test set: 6,538 images | 3D neutral/smile scans; multiview DSLR photos; verification pairs | `LICENSE`; institutional signature | Рабочий route, не open click-through. [NIST 3D-TEC record](https://tsapps.nist.gov/BDbC/Search/Details/366) подтверждает active status. |
| [WVU Twins Days 2010-2019](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/bme2.12090), США | 2,269 identities; paper reports 1,438 as identical-twin pairs/identities, wording ambiguous | Annual five-pose high-res DSLR; longitudinal IDs | `DUA` / FBI-NIST/WVU collaboration | Один из крупнейших longitudinal biometric twin resources; raw public download не найден. |
| [Japanese KAKEN 16H01752](https://kaken.nii.ac.jp/ja/grant/KAKENHI-PROJECT-16H01752/), Япония | Final report: 53 MZ pairs + 40 DZ/sibling pairs | Artec handheld 3D face scans, 23 landmarks, SNPs | `PAPER` / contact PI; [final report](https://kaken.nii.ac.jp/file/KAKENHI-PROJECT-16H01752/16H01752seika.pdf) | В гранте есть и Tohoku MRI, но это другая выборка: paired face+MRI утверждать нельзя. |
| [Osaka VISIA skin-aging twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC4141746/), Япония | 67 MZ pairs, age 40-87 | Standard, cross-polarized and UV facial images | `PAPER` / author request | Сильный face-age/skin-age сигнал; возможный, но не доказанный overlap с Osaka MRI registry. |
| [QIMR 2D facial morphology](https://publications.qimrberghofer.edu.au/Public/Publication/4160), Австралия | 1,320 twins and siblings | Standardized 2D face photographs + facial traits | `PAPER` / institute request | Тот же институт, что QTIM; ID overlap надо проверять отдельно. |
| [ALSPAC 3D twin subset](https://pubmed.ncbi.nlm.nih.gov/23323545/), Великобритания | 37 pairs: 19 MZ + 18 DZ, age 15 | 3D laser scans, 21 landmarks | `DUA` / ALSPAC | У ALSPAC также есть [neuroimaging resource](https://www.bristol.ac.uk/alspac/researchers/our-data/neuroimaging-data/); same-subject overlap неизвестен. |
| [Guyuron facial aging twins](https://pubmed.ncbi.nlm.nih.gov/19337100/), США | 186 MZ pairs | Digital facial images + lifestyle questionnaire | `PAPER` / author request | Хорошая aging hypothesis, но raw release не найден. |
| [Twins discordant for orofacial clefting](https://pmc.ncbi.nlm.nih.gov/articles/PMC5725745/), US/Denmark/Belgium | Final 44 unaffected co-twins: 10 MZ + 34 DZ, plus 241 controls | 3dMD surface + dense quasi-landmarks | `PAPER` / consortium contact | Полезно для craniofacial shape, но это не complete-pair general-population set. |
| [Marmara 3dMDface twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC10633807/), Турция | 14 MZ pairs | Full-face STL/3dMDface | `PAPER` | Малый validation set; raw archive не подтверждён. |
| [Brazilian photoanthropometry study](https://www.scielo.br/scielo.php?pid=S1516-18462026000101803&script=sci_arttext), Бразилия | 30 twin pairs, mostly MZ | Frontal/lateral 720p video frames + landmarks | `PAPER`; raw identifiable images not shared | Подтверждает локальную collection, но не является доступным image dataset. |
| [Samsung Medical Center Korean twin archive](https://pmc.ncbi.nlm.nih.gov/articles/PMC7719592/), Южная Корея | Source archive: 553 participants; analyzed subset 75 pairs: 36 MZ, 13 DZ and 26 same-sex sibling pairs | Lateral cephalograms and hard/soft-tissue craniofacial measures | `PAPER` / author or Samsung request | Несколько статей используют один более крупный source archive; запрос должен касаться полного image inventory, а не одной 13 MZ + 13 DZ публикации. |
| [Iranian cephalometric twins](https://wrap.warwick.ac.uk/id/eprint/55750/), Иран | Twin sample in study | Cephalograms + craniofacial landmarks | `PAPER` | Repository contains paper, not confirmed raw images. |
| [Sarajevo cephalometric twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC6643326/), Bosnia and Herzegovina | 52 pairs: 20 MZ + 32 DZ | Cephalograms, 19 dental/craniofacial variables | `PAPER` | Measurements reported; raw image access unknown. |
| [Small 3D laser twin study](https://pubmed.ncbi.nlm.nih.gov/15592212/) | 26 pairs: 10 MZ, 10 same-sex DZ, 6 opposite-sex DZ | 3D laser face scans | `PAPER` | Historical small set. |
| [3D stereophotogrammetry pilot](https://pmc.ncbi.nlm.nih.gov/articles/PMC3911821/) | 21 pairs: 10 MZ + 11 DZ | 3D surfaces + 13 landmarks | `PAPER` | Small but methods are reusable. |
| [Audio-Visual Twins Database](https://www.comp.nus.edu.sg/~face/paper/ICBA15.pdf), China/US/Russia | 39 twin pairs | Frontal/profile photos, facial-motion video, audio | `PAPER` / author request | Полезная multimodal biometric коллекция; public download не подтверждён. |
| [CASIA-IrisV4 Twins](https://hycasia.github.io/dataset/casia-irisv4/), Китай | 100 twin pairs | NIR iris images | `OPEN` direct 60 MB download from official page | Не лицо, но редкий реально открытый twin head biometric dataset. |
| [Iranian multimodal twin biometrics](https://doi.org/10.1109/IranianCEE.2013.6599528), Иран | 52 pairs / 104 people; 20 MZ pairs | 2D face, moving face, spectral and thermal face, fingerprint, handwriting | `PAPER` / author request | Редкая thermal/spectral lead, но download не найден. |

### 2a. Stereophotogrammetry, structured light and optical 3D

Этот подреестр включает строгую stereophotogrammetry и соседние optical 3D
методы. Последние помечены как laser/structured light, чтобы не смешивать разные
приборы. Ни одна из новых академических коллекций ниже не имеет подтверждённого
открытого скачивания raw identifiable surfaces; рабочий маршрут — запрос
владельцам архива.

| Исследование/архив | Размер | Прибор и данные | Повторы/соседние модальности | Доступ и ценность |
|---|---:|---|---|---|
| [Lithuanian Twin Centre / LSMU](https://www.cambridge.org/core/journals/twin-research-and-human-genetics/article/comparing-facial-3d-analysis-with-dna-testing-to-determine-zygosities-of-twins/45B1D3657B0A1C74BA4AA107610F4A15), Литва | 106 same-sex pairs recruited; 105 analyzed after one exclusion: 58 MZ + 47 DZ; mean age 20.5 | Portable 3dMD, textured dense surface, 21 manual landmarks, DNA zygosity | Repeat neutral captures for 83 people; a [derived chewing-side analysis](https://www.finna.fi/Record/jultika.nbnfi-fe2021112256324) used the same 106-pair source and retained 104 pairs | `PAPER` / contact LSMU. Самый сильный новый 3D-face target: крупный, DNA-typed, с repeatability data. |
| [LSMU craniofacial archive dissertation](https://lsmu.lt/cris/bitstream/20.500.12512/15954/2/Disertacija_Sidlauskas.pdf), Литва | 141 pairs: 90 MZ + 51 DZ, age 15.3-39.6 | 3D facial images and/or lateral head cephalograms | Registry >600 volunteer pairs; dental/orthodontic consultations and orthopantomograms mentioned | `PAPER` / institutional request. Нужна modality-overlap table: не у всех 141 пар гарантированы обе модальности. |
| [Twins Days FaceCam pilot](https://pmc.ncbi.nlm.nih.gov/articles/PMC3911821/), США | 21 child pairs: 10 MZ + 11 same-sex DZ, age 5-12 | FaceCam 250 structured-light stereophotogrammetry, >300k points, texture, 13 landmarks | One capture; medical history and zygosity questionnaire | `PAPER`; likely source archive behind the published study, raw access not stated. |
| [Marmara 3dMDface archive](https://pmc.ncbi.nlm.nih.gov/articles/PMC10633807/), Турция | 14 MZ pairs, mean age 14.75 | Six-camera 3dMDface, full surface exported to STL, regional RMS/deviation maps | Pretreatment orthodontic archive | `PAPER` / author request. Methods section clearly says pairs, despite abbreviated abstract wording. |
| [Marmara orthodontic co-twin pilot](https://katalog.marmara.edu.tr/veriler/yordambt/cokluortam/1E/FEBC75CA-BFB4-634D-B9BD-49DACA378D15.pdf), Турция | 14 genetic twin **individuals**; pair count not safely inferable | 3dMD facial scans | Pre/post treatment, dental casts and lateral cephalograms in the same archive | `PAPER` / thesis contact. Особенно интересен paired multi-modal и longitudinal design, но sample manifest обязателен. |
| [Milan VECTRA twin superimposition](https://pubmed.ncbi.nlm.nih.gov/29291466/), Италия | 10 MZ pairs, age 25-69 | VECTRA-3D M3 stereophotogrammetry, dense surface and RMS maps | Each participant captured twice; 220 self/twin/unrelated superimpositions | `PAPER` / author request. Хороший repeat-scan technical validation, маленький aging sample. |
| [Milan longitudinal infant twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC8869879/), Италия | One MZ pair | 3D stereophotogrammetric facial surfaces and 3D-3D superimposition | Eight timepoints from 6 months to 5 years | `PAPER`; case-level longitudinal reference, не training set. |
| [Inonu identical-twin case](https://abakus.inonu.edu.tr/items/eae6812c-832f-49fd-bcad-1ae373138440), Турция | One MZ pair, age 27 | 3dMD 3D photogrammetry, surface superimposition and color-distance map | One session; zygomatic, lip and cheek differences reported | `PAPER`; open article, but this is a case report and no reusable raw surface archive is offered. |
| [ALSPAC twin facial subset](https://onlinelibrary.wiley.com/doi/10.1111/ocr.12012), Великобритания | 37 pairs: 19 MZ + 18 DZ, age 15 | **Laser surface scan**, 21 landmarks and full-surface comparisons | Potential linkage to ALSPAC phenotypes/neuroimaging; overlap unconfirmed | `DUA` / ALSPAC. Не stereophotogrammetry, но методически близкий dense 3D target. |
| [Naini-Moss optical surface twin study](https://pubmed.ncbi.nlm.nih.gov/15592212/) | 26 pairs: 10 MZ, 10 same-sex DZ, 6 opposite-sex DZ | 3D optical/laser surface scanning | Facial regional comparisons | `PAPER`; historical archive, access route not published. |
| [Australian teeth-and-faces twin cohorts](https://www.cambridge.org/core/journals/twin-research-and-human-genetics/article/genetic-epigenetic-and-environmental-influences-on-dentofacial-structures-and-oral-health-ongoing-studies-of-australian-twins-and-their-families/4B8A3D488208C6F2778DF25341A079A8), Австралия | 1,293 pairs enrolled across three cohorts; face-imaged subset not tabulated in text | Cohort 1: synchronized Hasselblad stereo pairs in five views; later 2D/3D dental imaging | Longitudinal dental records, casts, photos; micro-CT of exfoliated teeth in later cohorts | `DUA` / Adelaide contact. Не утверждать, что все 1,293 пары имеют stereo faces. |
| [Pakistan/Wateen 3dMD study](https://medicalforummonthly.com/index.php/mfm/article/view/4347), Пакистан | Article reports 20 MZ pairs, mean age 5 +/- 1 | 3dMDface, facial thirds and surface deviation | No linked modalities reported | `PAPER`, **unverified lead**: до использования запросить ethics record, pair manifest and raw-data policy. |
| [Historical serial facial-contour study](https://www.cambridge.org/core/journals/amg-acta-geneticae-medicae-et-gemellologiae-twin-research/article/intrapair-facial-differences-in-twins/43FEEC1A6BE10D32FE449A0164E0B0DF) | 18 like-sexed pairs, near-equal MZ/DZ distribution | Serial stereophotogrammetric facial contour maps | Annual records | `PAPER`; archival lead, digital raw data may not exist. |
| [Orofacial-cleft co-twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC5725745/) | 44 unaffected co-twins + 241 controls | 3dMDface and spatially dense quasi-landmarks | Pittsburgh, Denmark, Belgium sources | `PAPER` / consortium request; useful craniofacial outlier cohort, not complete-pair general population. |

## 3. Dental, palate, retina and OCT

| Ресурс | Размер | Изображения/3D | Доступ | Вывод |
|---|---:|---|---|---|
| [EMIF-AD PreclinAD retinal-MRI twins](https://pubmed.ncbi.nlm.nih.gov/30326071/), Нидерланды | 134 cognitively healthy people / 67 complete MZ pairs age 60+ in direct analysis; broader Amsterdam baseline has 100 complete pairs | Two Topcon digital fundus fields/eye, Heidelberg Spectralis OCT, fundus autofluorescence; paired structural/functional MRI, PET and MEG | `DUA` via [EMIF-AD](https://www.emif.eu/emif-ad-2/) | **Лучший подтверждённый twin fundus + MRI dataset.** Published analysis used source fundus-derived vascular traits and MRI WMH volumes. |
| [Monash/ATR retinal OCT + MRI twins](https://www.nature.com/articles/s41598-022-13662-8), Австралия | 35 people / 18 pairs, T2D-discordant recruitment | Bilateral Heidelberg OCT, retinal photographs; paired 3T T1, FLAIR, DTI, SWI and ASL | `DUA`; reasonable request to corresponding author | **Прямой paired dataset.** Raw images not public; ask whether E2E OCT, photographs and source MRI are retained and shareable. |
| [3D intraoral twin scans](https://pmc.ncbi.nlm.nih.gov/articles/PMC7532102/) | 64 MZ + 33 same-sex DZ + 7 opposite-sex DZ participants as reported | Three repeated intraoral scans; palate/rugae STL | `PAPER` | Хороший repeatability/shape resource; raw STL не опубликованы. |
| [Digital dental casts, 2024](https://www.nature.com/articles/s41598-024-70985-4) | 50 MZ + 35 DZ pairs | Digital dental casts and tooth-size phenotypes | `PAPER` | Более крупный dental twin target. |
| [344 digitized dental casts](https://pubmed.ncbi.nlm.nih.gov/27350700/) | 86 twin pairs | 344 3D dental models | `PAPER` | Повторные модели потенциально полезны для within-pair geometry. |
| [Guangzhou Twin Eye Study](https://www.cambridge.org/core/journals/twin-research-and-human-genetics/article/guangzhou-twin-eye-study-2019-update/90C6A3D2A49870EAE5B24D73E37F1263), Китай | ~1,300 twin pairs | Fundus, HRT-3, AS-OCT, SS-OCT, SD-OCT | `DUA` / collaboration; [dataset atlas](https://atlaslongitudinaldatasets.ac.uk/datasets/gtes) | Сильнейшая азиатская longitudinal eye-imaging twin cohort. |
| [TwinsUK macular OCT](https://pmc.ncbi.nlm.nih.gov/articles/PMC7405717/) | 184 adult twins | 3D macular OCT | `DUA` via TwinsUK | Проверить overlap с 3D-face и новой MRI cohort. |
| [Tasmanian Eye Study of Twins](https://pmc.ncbi.nlm.nih.gov/articles/PMC3855231/) | Twin cohort; modality-specific N varies | Stereo fundus, vessels, optic disc, conjunctival autofluorescence, UV photos | `DUA` / study contact | Ещё один необычный visual-aging channel; images не заявлены как open. |

## 4. Коммерческие и лицензируемые данные

| Ресурс | Что реально продаётся/выдаётся | Twin signal | Статус и ограничение |
|---|---|---|---|
| [Qlean `ds-016`](https://qleandataset.visual-bank.co.jp/en/lineup/ds-016), Япония | 67,224 PNG/JPG, 468 people, age 7 to 50+, lighting/pose/accessories/expression, 57 GB | 12 **individuals** marked identical twins; complete-pair count не указан | `COMMERCIAL`; commercial/research use, rights-cleared, free sample request; separate [academia program](https://qleandataset.visual-bank.co.jp/academia). |
| [Qlean walking-video collection](https://qleandataset.visual-bank.co.jp/news/20250819), Япония | 2,820 MP4, 470 Japanese participants | 14 MZ twin individuals | `COMMERCIAL`; полезно как gait/body adjunct, не head scan. |
| [Datatang 23 identical-twin pairs](https://paperswithcode.com/dataset/23-pairs-of-identical-twins-face-image-data), Китай | 40 images/person, indoor/outdoor, age groups | 23 MZ pairs | `COMMERCIAL`, но ссылка подтверждена только независимой карточкой; требовать vendor sample, manifest и quote до оплаты. |
| [PIXTA Japanese face dataset](https://pixta.co.jp/news/1876) | 1,000 rights-cleared images / 200 people, age bins; optional face box/age/sex labels | Twin labels отсутствуют | `COMMERCIAL`; японский demographic control. |
| [Appen human-behavior data](https://www.appen.com/human-behaviour-ai-data) | Custom consented face/gaze/expression/full-body collection | Можно заказать twin recruitment, готовый twin MRI product не заявлен | `COMMERCIAL`; RFQ/custom collection, не off-the-shelf dataset. |
| [HumanScanRepository](https://humanscanrepository.com/product/hsr0235-head-au17/) | Individual high-resolution textured 3D head products with commercial license | Twin labels отсутствуют | `COMMERCIAL`; geometry/render pretraining only. |

**Не найдено:** ни одной проверяемой коммерческой карточки twin MRI, twin CT или
twin CBCT. Реалистичный коммерческий путь здесь — заказать consented collection
у CRO/data vendor либо лицензировать face twin data и отдельно договариваться с
академическим registry о MRI. Это не даёт paired IDs автоматически.

## 5. Не близнецы, но парные face/head <-> brain данные

| Ресурс и регион | Парные данные | Доступ | Зачем нужен |
|---|---|---|---|
| [UK Biobank eye-brain imaging](https://www.nature.com/articles/s41467-024-50309-w), Великобритания | Same-person fundus features, OCT and multimodal brain MRI; 2024 study used average n=6,454 for discovery and 959 for independent validation | `DUA` via [UK Biobank imaging access](https://www.ukbiobank.ac.uk/about-our-data/types-of-data/imaging-data/) | Самый крупный доступный non-twin eye-brain benchmark; source OCT acquisition includes a magnified fundus photograph. |
| [ARIC retinal photography + brain MRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC3150229/), США | Carotid MRI substudy: 2,066 examined; 1,120 analyzed with retinal photography at two visits and brain MRI | `DUA` through [NHLBI BioLINCC](https://biolincc.nhlbi.nih.gov/studies/?q=ARIC&s=-available_resources) / dbGaP; verify availability of source images | Large longitudinal vascular-aging target with repeat fundus imaging, MRI and extensive cardiovascular covariates. |
| [KaiLuan eye-brain cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770952/), Китай | 908 participants underwent fundus photography and brain MRI in 2021-2022 | `PAPER` / author request | Strong East-Asian paired control for retinal vascular age, cognition and cerebral small-vessel disease; raw-image route must be confirmed. |
| [Dunedin Study](https://www.nature.com/articles/s43587-025-00897-z), Новая Зеландия | В age-45 phase: brain MRI и standardized non-smiling face photos с independent perceived-age ratings; ~93% участников фазы прошли MRI | `DUA` / concept paper; [official MRI overview](https://dunedinstudy.otago.ac.nz/studies), [data-use guide](https://sites.duke.edu/moffittcaspiprojects/data-use-guidelines/) | Наиболее прямой найденный face-age vs brain-age validation set. |
| [Lothian Birth Cohort 1936](https://pmc.ncbi.nlm.nih.gov/articles/PMC6225560/), Шотландия | 814 had bilateral fundus images at wave 2; 603 retained in a vessel analysis; brain MRI and later facial photographs/perceived age exist in the cohort | `DUA`; [data-request route and imaging dictionary](https://lothian-birth-cohorts.ed.ac.uk/data-access-collaboration); exact cross-modality overlap must be requested | Rare longitudinal bridge among retina, brain MRI, cognition and later facial age, but visit timing and raw-image overlap must be audited. |
| [BioFace3D](https://www.sciencedirect.com/science/article/pii/S0169260725004274), Испания | Open pipeline строит 3D face mesh и landmarks из MRI; BF3D has 6 consented scans; separate proprietary validation has 234 MRI | Code open; images proprietary | Практический baseline именно для нашей задачи; [paper PDF](https://dau.url.edu/bitstream/handle/20.500.14342/5489/BioFace3D.pdf?isAllowed=y&sequence=1). |
| [NoW](https://now.is.tue.mpg.de/dataset.html) | 2,054 phone photos of 100 subjects + separate ground-truth 3D head scan/person | Research license; validation scans/landmarks available | Проверка 2D -> 3D head reconstruction, не brain-age. |
| [INCRAN](https://papers.miccai.org/miccai-2025/0601-Paper2543.html) | Infant 3D photogrammetry plus smaller CT-derived head/skull pairs and dense feature points | Trained model/code planned; **raw patient data cannot be released** | Важный paired CT/soft-tissue analogue, но прежнее обозначение «public dataset» было неверным. |
| [RenderMe-360](https://renderme-360.github.io/inner-download.html), Китай | 500 IDs, 60 synchronized 2K cameras, 3D scans, 2D/3D landmarks, FLAME fits | `LICENSE` / research request | Сильный 4D/3D head pretraining set; twins not labeled. |
| [AST-Face](https://www.nature.com/articles/s41597-026-07098-2), East Asia | 98 topology-standardized 3D faces; 52 with synchronized multiview RGB/textured meshes | Processed public + identifiable subset under DUA | Азиатский 3D geometry/expression control, не twin set. |
| [BrainLat](https://pmc.ncbi.nlm.nih.gov/articles/PMC10710425/), Argentina/Chile/Colombia/Mexico/Peru | 780 participants: anatomical MRI, rs-fMRI, DWI, high-density EEG, clinical/cognitive data | Managed access | Не лицо и не twins, но лучший найденный Latin-American neuroimaging control. |
| [Nigerian Clinical MRI](https://www.nature.com/articles/s41597-025-04743-0), Нигерия | 88 participants, 761 T1/T2/FLAIR sequences | `DUA`; distributed images brain-masked/defaced | Африканский MRI control only; не подходит для лица и не содержит twin labels. |
| [UCT/MRC FAS photogrammetry platform](https://www.isprs.org/proceedings/xxxiv/part3/papers/paper051.pdf), ЮАР | Simultaneous stereo photographs of 6-7-year-old children; cohort size not stated in the methods paper | `PAPER`; two calibrated high-resolution cameras, 3D eye landmarks, no raw-data route found | Подтверждает африканскую facial stereophotogrammetry infrastructure, но это не twins и не открытый dataset. |

## 6. Региональные пробелы

| Регион | Что удалось подтвердить | Что не найдено |
|---|---|---|
| Китай | CATS MRI/fMRI twins; Guangzhou eye imaging; CASIA iris; commercial Datatang lead | Public non-defaced twin T1/CT with complete face. |
| Тайвань | Отдельные craniofacial/medical imaging papers встречаются, но проверяемой twin head-image cohort не найдено | Twin MRI/CT/3D-face dataset с маршрутом доступа. |
| Япония | Osaka MRI/PET/MEG + dentistry/dermatology; KAKEN 3D face; Qlean commercial twin subset | Подтверждённый same-subject pair `3D face + MRI`. |
| Южная Корея | Korean cephalometric twin paper и general twin registry | Raw cephalograms или MRI twin release. |
| Латинская Америка | TwinsMX MRI/fMRI/MRS; Brazilian photo study; BrainLat controls | Public twin MRI-face/CT-face dataset. |
| Африка | Nigerian MRI control; [Nigerian Twin and Sibling Registry](https://pmc.ncbi.nlm.nih.gov/articles/PMC10822790/) exists; UCT/MRC built a calibrated child-face stereo-photogrammetry platform | Какая-либо подтверждённая African **twin** head-imaging collection or a current raw-data route for the UCT images. |

## 7. Link audit: что исключено или понижено

| Ссылка/утверждение | Результат проверки | Решение |
|---|---|---|
| DLFR 440-image dataset | [Paper](https://www.sciencedirect.com/science/article/pii/S0045790621002986) claims public data/code, но заявленный site `sshmousavi.github.io/dlfr/` выглядит как незаполненный шаблон, download links не работают | `PAPER`; писать автору, не давать site как download. |
| OpenDataLab/Payititi twin cards | Карточки пустые, требуют JS/login либо исчезают; состав и лицензия не проверяются независимо | Исключены из рабочего shortlist. |
| Datatang 52-pair direct page | Vendor URL нестабилен/не отдаёт проверяемое описание | Не включён; 23-pair lead оставлен только с явной оговоркой и sample gate. |
| OpenNeuro dynamic dataset pages | Иногда выглядят пустыми в browser/parser | Для QTIM/QTAB даны стабильные GitHub DataLad mirrors плюс peer-reviewed data papers. |
| INCRAN as “public paired CT+3D dataset” | Авторы прямо сообщают, что raw infant data нельзя выпустить | Исправлено на paper/model-only resource. |
| Weizmann Project 10K as paired 3D-face+MRI | Public project page подтверждает deep phenotyping, но не удалось подтвердить обе модальности и same-subject linkage | Не включён до получения data dictionary. |

## 8. Минимальный запрос владельцам данных

До подачи полной заявки стоит получить письменные ответы на семь бинарных
вопросов:

1. Есть ли таблица complete MZ/DZ pairs и stable family/participant IDs?
2. Есть ли original T1 DICOM/NIfTI до defacing и skull stripping?
3. Входит ли лицо от forehead до chin, а не только верхняя часть носа/орбит?
4. Разрешает ли consent/DUA реконструкцию facial surface и biometric embeddings?
5. Есть ли photos, 3D face, VISIA, dental, OCT или video у **тех же** IDs?
6. Есть ли exact acquisition date, chronological age и повторные visits?
7. Разрешены ли model training, derived landmarks/meshes и публикация aggregate results?

Без положительных ответов на пункты 2-4 MRI-когорту следует считать brain-only,
даже если её T1 формально доступен.
