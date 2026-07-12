# Multimodal Feature Possibility Map

Date: 2026-07-12

## Purpose

This document expands the starter health-feature palette into a systematic
possibility map. The goal is to enumerate feature families until the project
hits a principled "not measurable from these data" boundary.

This is a research and cohort-design map. It is not a diagnostic protocol and
not a claim that structural brain MRI can infer every disease.

## Stop Rule

For each proposed feature family, ask:

1. Is the target anatomical structure, physiological process, behavior, lab
   marker, or exposure present in the data?
2. If not directly present, is there a known and measurable proxy with a
   plausible mechanism?
3. Is there an external label or outcome that can validate the proxy?
4. Can age, sex, scanner/site, preprocessing, total intracranial volume, and
   socioeconomic or clinical confounders be handled?
5. Would the claim still make sense for n=1, or only for population cohorts?

If the answer is "no" at steps 1-3, the row should move to `stop_invalid` or
`needs_other_modality`, not remain a model target.

## Evidence Tiers

- `tier_1_direct`: directly measurable if the named modality exists and QC
  passes.
- `tier_2_indirect`: usable as an interpretable proxy, but not a diagnosis.
- `tier_3_exploratory`: plausible research hypothesis; requires external labels
  and replication.
- `tier_4_needs_other_modality`: important for the palette but not inferable
  from the current brain MRI branch.
- `tier_5_stop_invalid`: do not model from current data; use as an explicit
  boundary example.

## Full Feature Inventory

This is the canonical markdown inventory. It replaces the earlier tabular
working draft.

### tier_1_direct

- `brain_structure / global_brain_size`: TIV,total brain volume,brain parenchymal fraction. Modality: `T1_MRI`. Valid use: confound control,atrophy context,sex shortcut audit. Invalid claim: global size is a health score. Next data: segmentation,morphometry,QC.
- `brain_structure / tissue_compartments`: GM,WM,CSF volumes. Modality: `T1_MRI`. Valid use: structural aging and atrophy profile. Invalid claim: tissue volumes diagnose disease alone. Next data: segmentation,morphometry,age/sex/site covariates.
- `brain_structure / cortical_morphometry`: cortical thickness,surface area,curvature,gyrification. Modality: `T1_MRI`. Valid use: regional structural phenotype and aging slopes. Invalid claim: one cortical region proves a disease. Next data: surface pipeline,QC,reference cohort.
- `brain_structure / subcortical_volumes`: hippocampus,amygdala,thalamus,caudate,putamen,pallidum. Modality: `T1_MRI`. Valid use: neurodegeneration and psychiatric-risk covariance. Invalid claim: single volume is a diagnosis. Next data: segmentation,ICV adjustment,normative cohort.
- `brain_structure / ventricles_and_csf`: lateral ventricles,third ventricle,sulcal CSF. Modality: `T1_T2_MRI`. Valid use: atrophy,hydrocephalus-screening context,pressure hypothesis triage. Invalid claim: ventricle size directly measures pressure. Next data: T2/FLAIR,clinical exam,ophthalmology if pressure question.
- `brain_structure / medial_temporal_lobe`: hippocampal volume,entorhinal thickness,temporal horn. Modality: `T1_MRI`. Valid use: AD-like atrophy pattern research. Invalid claim: AD diagnosis from T1 alone. Next data: cognition,amyloid/tau/PET/CSF if AD claim.
- `brain_structure / cerebellum_brainstem`: cerebellar volume,brainstem volume,midbrain measures. Modality: `T1_MRI`. Valid use: motor,ataxia,neurodegeneration hypothesis features. Invalid claim: Parkinson or ataxia diagnosis from volume alone. Next data: clinical labels,DAT/NM-MRI where relevant.
- `brain_structure / asymmetry_laterality`: left-right volume/thickness asymmetry. Modality: `T1_MRI`. Valid use: developmental,vascular,lesion and QC hypothesis marker. Invalid claim: personality or talent inference. Next data: normative cohort,handedness,lesion labels.
- `brain_structure / incidental_gross_abnormalities`: mass effect,large lesion,malformation,large cyst. Modality: `MRI_visible_field`. Valid use: flag for radiology review only. Invalid claim: automated diagnosis without radiologist. Next data: clinical radiology workflow.
- `vascular_brain / white_matter_hyperintensities`: WMH volume,Fazekas-like burden,periventricular/deep WMH. Modality: `FLAIR_T2`. Valid use: small-vessel disease burden and vascular brain-age feature. Invalid claim: WMH from T1-only is validated. Next data: FLAIR/T2,WMH segmentation,vascular risk labels.
- `vascular_brain / lacunes_and_old_infarcts`: lacunes,cortical infarcts,strategic infarcts. Modality: `T1_T2_FLAIR`. Valid use: vascular injury marker. Invalid claim: stroke timing or cause from T1 alone. Next data: FLAIR,DWI,clinical history.
- `vascular_brain / perivascular_spaces`: basal ganglia PVS,centrum semiovale PVS. Modality: `T2_highres`. Valid use: SVD and glymphatic hypothesis marker. Invalid claim: PVS count proves glymphatic dysfunction. Next data: T2,standardized rating,age/vascular risk.
- `vascular_brain / microbleeds_siderosis`: microbleed count,distribution,superficial siderosis. Modality: `SWI_T2star_GRE`. Valid use: hemorrhagic SVD/CAA risk phenotype. Invalid claim: absence on T1 excludes microbleeds. Next data: SWI/T2*,validated detector.
- `vascular_brain / acute_diffusion_lesions`: DWI positive lesions,acute infarcts. Modality: `DWI_ADC`. Valid use: acute/subacute ischemia marker in clinical workflow. Invalid claim: DWI lesion cause without clinical context. Next data: DWI/ADC,clinical/radiology review.
- `vascular_brain / perfusion`: CBF,arterial transit time,perfusion asymmetry. Modality: `ASL_or_perfusion_MRI`. Valid use: vascular reserve and neurodegeneration covariate. Invalid claim: perfusion equals cognition or pressure. Next data: ASL,vascular risk,cognition.
- `vascular_brain / large_vessel_lumen`: stenosis,aneurysm,tortuosity. Modality: `MRA_CTA`. Valid use: vascular anatomy and risk phenotype. Invalid claim: T1 brain scan excludes stenosis/aneurysm. Next data: MRA/CTA,vascular labels.
- `vascular_brain / vessel_wall`: wall thickening,enhancement,plaque. Modality: `vessel_wall_MRI`. Valid use: intracranial atherosclerosis/inflammation phenotype. Invalid claim: wall disease from ordinary T1. Next data: vessel-wall protocol,clinical labels.
- `white_matter_microstructure / dti_metrics`: FA,MD,RD,AD. Modality: `DWI_DTI`. Valid use: white-matter integrity and white-matter brain-age feature. Invalid claim: tract density from T1. Next data: DWI,eddy/QC,tract atlas.
- `white_matter_microstructure / fixel_metrics`: FD,FC,FDC. Modality: `multi_shell_DWI`. Valid use: fiber-specific density/cross-section research. Invalid claim: fixel density from standard T1. Next data: multi-shell/high-b DWI,MRtrix pipeline.
- `white_matter_microstructure / noddi_free_water`: NDI,ODI,FISO,free-water. Modality: `multi_shell_DWI`. Valid use: neurite/free-water hypothesis features. Invalid claim: hydration status from one T1. Next data: multi-shell DWI,model QC.
- `qmri / myelin_water`: MWF,MPF,MTsat. Modality: `qMRI_MWF_MT`. Valid use: myelin aging/neurodegeneration marker. Invalid claim: myelin content from ordinary T1 intensity. Next data: qMRI protocol,phantom/QC.
- `qmri / iron_susceptibility`: QSM,R2*,basal ganglia iron. Modality: `QSM_R2star`. Valid use: iron/aging/neurodegeneration marker. Invalid claim: iron from T1 alone. Next data: QSM/SWI pipeline.
- `sex_hormones / sex_as_moderator`: known sex,predicted sex probability,sex interactions. Modality: `labels_MRI_model_outputs`. Valid use: fairness,shortcut,QC,sex-stratified risk architecture. Invalid claim: male/female brain is healthier. Next data: known labels,TIV,age,site.
- `retina_oculomics / retinal_age`: fundus age,OCT age,OCT age gap. Modality: `fundus_OCT`. Valid use: noninvasive aging/systemic vascular feature. Invalid claim: retinal age diagnoses disease alone. Next data: retina images,outcome labels.
- `retina_oculomics / retinal_microvasculature`: vessel caliber,tortuosity,OCTA density. Modality: `fundus_OCTA`. Valid use: vascular/metabolic risk phenotype. Invalid claim: retinal vessels replace lipid/BP labs. Next data: fundus/OCTA,vascular labels.
- `retina_oculomics / retinal_neurodegeneration`: RNFL,GCL,macular thickness,optic disc. Modality: `OCT`. Valid use: sensory/neurodegeneration covariate. Invalid claim: OCT proves brain dementia. Next data: OCT,vision,cognition.
- `body_organ_mri / adiposity_distribution`: visceral fat,subcutaneous fat,liver fat,muscle fat. Modality: `body_MRI_DXA`. Valid use: metabolic aging and cardiometabolic risk. Invalid claim: visceral fat from brain MRI. Next data: body MRI/DXA.
- `body_organ_mri / cardiac_structure_function`: LV mass,EF,atrial volume,aortic stiffness. Modality: `cardiac_MRI_ECG`. Valid use: heart-brain aging and vascular risk. Invalid claim: cardiac age from brain MRI. Next data: CMR,ECG,BP.
- `body_organ_mri / liver_pancreas_kidney_spleen`: organ volume,fat,iron,cysts. Modality: `abdominal_MRI_labs`. Valid use: multi-organ aging profile. Invalid claim: abdominal disease from brain scan. Next data: abdominal MRI,labs.
- `robustness_uncertainty / test_retest_stability`: ICC,CV,within-subject SD. Modality: `repeat_scans`. Valid use: reproducibility gate. Invalid claim: stable output proves clinical truth. Next data: test-retest data.
- `robustness_uncertainty / site_scanner_sensitivity`: site fixed effects,travelling-subject variance. Modality: `multisite_scans`. Valid use: domain shift and portability assessment. Invalid claim: site stability validates biology. Next data: travelling subjects.
- `robustness_uncertainty / preprocessing_perturbation_delta`: skull-strip,resample,rotation,brain-size perturbation deltas. Modality: `pipeline_variants`. Valid use: model robustness and QC. Invalid claim: perturbation robustness proves accuracy. Next data: paired perturbation runs.

### tier_2_indirect

- `qmri / water_hydration`: T2,T2*,PD,free-water proxies. Modality: `qMRI_T2_DWI`. Valid use: tissue water/free-water phenotype. Invalid claim: daily hydration from structural MRI. Next data: qMRI/DWI,hydration/lab context.
- `pressure_csf / intracranial_pressure_proxies`: optic nerve sheath,empty sella,venous sinus stenosis,ventricles. Modality: `orbital_MRI_MRV_T2`. Valid use: triage signs for raised pressure hypothesis. Invalid claim: true intracranial pressure estimate. Next data: ophthalmology,MRV,clinical exam,LP if clinically indicated.
- `neurodegeneration / ad_like_pattern`: MTL atrophy,posterior cortical atrophy,WMH context. Modality: `T1_FLAIR`. Valid use: risk-pattern feature with cognition labels. Invalid claim: Alzheimer diagnosis from MRI alone. Next data: cognitive tests,amyloid/tau/PET/CSF.
- `neurodegeneration / ftd_like_pattern`: frontal/anterior temporal atrophy,asymmetry. Modality: `T1_MRI`. Valid use: FTD-pattern hypothesis feature. Invalid claim: FTD diagnosis from T1 alone. Next data: clinical labels,language/behavior tests.
- `neurodegeneration / parkinson_lbd_axis`: nigrosome,NM substantia nigra,basal ganglia,olfaction,RBD. Modality: `NM_MRI_SWI_DTI_clinical`. Valid use: PD/LBD hypothesis when correct modalities exist. Invalid claim: NeuroFM sex/age output diagnoses PD/LBD. Next data: neurology labels,DAT/NM-MRI,RBD/olfaction.
- `neurodegeneration / ms_demyelination_axis`: lesion count,location,black holes,atrophy. Modality: `FLAIR_T2_T1_Gd`. Valid use: demyelination burden if MS protocol exists. Invalid claim: MS diagnosis from noncontrast T1 alone. Next data: MS protocol,radiology,clinical criteria.
- `neurodegeneration / motor_neuron_axis`: motor cortex thickness,corticospinal FA. Modality: `T1_DWI`. Valid use: ALS/motor-system hypothesis feature. Invalid claim: ALS diagnosis from motor cortex size. Next data: DWI,EMG/clinical labels.
- `sensory_reserve / hearing_axis`: audiometry,speech-in-noise,auditory cortex,temporal lobe. Modality: `audiometry_T1_fMRI`. Valid use: hearing reserve as dementia/health moderator. Invalid claim: auditory cortex morphology proves hearing loss. Next data: audiometry,hearing aid use,cognition.
- `sensory_reserve / vision_axis`: visual acuity,OCT RNFL/GCL,occipital cortex,optic radiation. Modality: `vision_tests_OCT_MRI`. Valid use: vision reserve and neurodegeneration/vascular covariate. Invalid claim: visual cortex proves eyesight. Next data: OCT/fundus,acuity,ophthalmology labels.
- `sensory_reserve / olfaction_axis`: olfactory bulb,smell test,ENT context. Modality: `olfaction_test_MRI`. Valid use: PD/AD/ENT hypothesis covariate. Invalid claim: smell ability from brain T1 alone. Next data: smell tests,ENT,clinical labels.
- `pain_stress / chronic_pain_axis`: insula,ACC,PFC,thalamus,connectivity,pain questionnaires. Modality: `fMRI_T1_questionnaires`. Valid use: mechanism/risk phenotype in cohorts. Invalid claim: brain MRI proves pain level. Next data: validated pain scales,diagnosis,longitudinal labels.
- `pain_stress / allostatic_load_axis`: BP,waist,HbA1c,CRP,cortisol,HRV,sleep,hippocampus. Modality: `labs_wearables_MRI`. Valid use: cumulative stress physiology profile. Invalid claim: stress diagnosis from hippocampus alone. Next data: labs,cortisol/wearables,questionnaires.
- `psychiatric / depression_anxiety_sleep`: PHQ/GAD/sleep scales,default mode,limbic volumes,actigraphy. Modality: `questionnaires_wearables_MRI`. Valid use: population association and confound control. Invalid claim: mental disorder diagnosis from T1. Next data: validated scales,clinical labels,sleep data.
- `psychiatric / reward_apathy_anhedonia`: nucleus accumbens,OFC,ACC,SHAPS,apathy scales. Modality: `questionnaires_fMRI_T1`. Valid use: reward/apathy phenotype with labels. Invalid claim: pleasure center score from structure alone. Next data: scales,behavior/fMRI,diagnoses.
- `psychosocial / loneliness_social_support`: UCLA loneliness,network size,social support,DMN/MTL features. Modality: `questionnaires_behavior_MRI`. Valid use: social health covariate and outcome association. Invalid claim: loneliness from MRI alone. Next data: validated loneliness/social scales.
- `mobility_environment / life_space_mobility`: radius of gyration,location entropy,home-stay,outdoor time. Modality: `GPS_wearable_environment`. Valid use: behavioral phenotype and exposure layer. Invalid claim: small radius proves loneliness. Next data: privacy-preserving GPS,questionnaires.
- `mobility_environment / green_blue_exposure`: NDVI,park time,water proximity,outdoor daylight. Modality: `GPS_environmental_data`. Valid use: exposome covariate for cognition/mood/vascular health. Invalid claim: park exposure cures dementia. Next data: geospatial data,weather,SES.
- `mobility_environment / pollution_noise_heat`: PM2.5,NO2,noise,heat,urban density. Modality: `geospatial_exposure`. Valid use: environmental risk covariate. Invalid claim: brain MRI identifies air pollution exposure precisely. Next data: residential/GPS exposure models.
- `sex_hormones / menopause_reproductive_transition`: menopause status,HRT,cycle,pregnancy history. Modality: `questionnaire_EHR_labs`. Valid use: vascular/neurodegeneration interaction analysis. Invalid claim: menopause status from T1 alone. Next data: questionnaires,hormones,EHR.
- `cardiometabolic / blood_pressure_vascular_risk`: BP,hypertension meds,arterial stiffness. Modality: `clinical_measures`. Valid use: vascular brain risk covariates. Invalid claim: BP from brain MRI. Next data: measured BP,medications.
- `skin_face / face_age`: face age,wrinkles,pigmentation,periorbital features. Modality: `face_photos_skin_images`. Valid use: appearance-age and health association. Invalid claim: face age is clinical diagnosis. Next data: standardized photos,labels,ethics.
- `skin_face / skin_inflammation_barrier`: eczema,psoriasis,skin texture,lesions. Modality: `dermatology_images_exam`. Valid use: inflammatory phenotype if dermatology data exists. Invalid claim: skin disease from brain MRI. Next data: derm images/exam.
- `craniofacial_skull / skull_craniofacial_shape`: skull thickness,craniofacial ratios,jaw relation. Modality: `head_MRI_CT_CBCT`. Valid use: developmental/anatomical covariate. Invalid claim: skull bones diagnose posture or health. Next data: CT/CBCT,orthodontic/posture labels.
- `dental_oral / dental_age`: pulp-to-tooth ratio,secondary dentin,tooth loss. Modality: `dental_Xray_CBCT`. Valid use: dental age/oral aging feature. Invalid claim: tooth age from brain MRI. Next data: dental imaging.
- `dental_oral / periodontal_inflammation`: pocket depth,bone loss,bleeding,gum inflammation. Modality: `dental_exam_Xray`. Valid use: oral inflammation and vascular/dementia covariate. Invalid claim: gum inflammation from brain MRI. Next data: periodontal exam,dental Xray.
- `ent_sinus / sinus_mucosal_burden`: mucosal thickening,cysts,polyps,opacification. Modality: `head_MRI_CT_sinus`. Valid use: ENT/inflammation covariate with symptoms. Invalid claim: incidental thickening diagnoses sinusitis. Next data: symptoms,SNOT-22,ENT/CT.
- `ent_sinus / nasal_airway`: septal deviation,turbinate hypertrophy,airway volume. Modality: `CT_MRI_ENT`. Valid use: sleep/ENT covariate. Invalid claim: airway anatomy proves sleep apnea. Next data: ENT exam,sleep study.
- `sleep_breathing / sleep_apnea_axis`: AHI,oxygen desaturation,airway,WMH,hypertension. Modality: `polysomnography_wearable_ENT_MRI`. Valid use: vascular/cognition risk covariate. Invalid claim: sleep apnea from brain MRI. Next data: sleep study,wearables,ENT.
- `musculoskeletal / posture_scoliosis`: Cobb angle,sagittal balance,cervical lordosis,forward head. Modality: `spine_Xray_EOS_3D_photo`. Valid use: posture/spine phenotype. Invalid claim: scoliosis from skull alone. Next data: spine imaging,posture assessment.
- `musculoskeletal / sarcopenia_frailty`: muscle volume,fat infiltration,grip strength,gait speed. Modality: `body_MRI_DXA_wearables`. Valid use: frailty and metabolic aging feature. Invalid claim: sarcopenia from brain MRI. Next data: body MRI/DXA,strength tests.
- `musculoskeletal / bone_health`: BMD,vertebral fractures,skull/bone marrow signals. Modality: `DXA_CT_MRI`. Valid use: bone aging/frailty covariate. Invalid claim: osteoporosis from brain MRI skull signal. Next data: DXA/CT,fracture history.
- `wearables / physical_activity`: steps,MVPA,sedentary time,gait speed. Modality: `accelerometer_wearable`. Valid use: activity/frailty/vascular covariate. Invalid claim: activity level from structural MRI. Next data: wearable data.
- `wearables / sleep_circadian`: sleep duration,fragmentation,chronotype,rhythm stability. Modality: `wearable_sleep_diary`. Valid use: sleep and inflammation/aging covariate. Invalid claim: sleep quality from brain MRI. Next data: wearables,sleep diary.
- `wearables / autonomic_cardiovascular`: HRV,resting HR,PPG age,VO2max proxy. Modality: `wearable_ECG_CPET`. Valid use: cardiorespiratory reserve feature. Invalid claim: HRV from MRI. Next data: wearable/ECG/CPET.
- `lifestyle / smoking_alcohol_diet`: pack-years,alcohol,diet quality,caffeine. Modality: `questionnaire_EHR`. Valid use: confounders and modifiable risk layer. Invalid claim: lifestyle reconstructed from MRI. Next data: questionnaires,EHR.
- `lifestyle / medications`: statins,antihypertensives,HRT,antidepressants. Modality: `EHR_questionnaire`. Valid use: confound control and treatment modifiers. Invalid claim: medication use from MRI. Next data: EHR/medication list.

### tier_3_exploratory

- `pressure_csf_glymphatic / glymphatic_clearance_axis`: DTI-ALPS,PVS burden,sleep fragmentation,CSF-space geometry. Modality: `DWI_T2_sleep_data`. Valid use: exploratory clearance/sleep/SVD hypothesis. Invalid claim: glymphatic function proven from one MRI feature. Next data: DWI,T2,sleep data,replication.
- `pain_stress / pain_network_signature`: multivariate pain signatures,connectivity,radiomics,task response. Modality: `fMRI_task_rest_MRI`. Valid use: candidate mechanism or treatment-response biomarker. Invalid claim: individual legal/clinical proof of pain. Next data: standardized tasks,external validation.
- `psychiatric / psychiatric_network_embeddings`: DMN,salience,limbic connectivity,foundation embeddings. Modality: `fMRI_T1_foundation_model`. Valid use: exploratory phenotype clustering with labels. Invalid claim: psychiatric diagnosis from embeddings. Next data: validated psychiatric labels,longitudinal replication.
- `psychiatric / reward_network_signature`: ventral striatum,OFC,ACC,functional connectivity. Modality: `fMRI_behavior_scales`. Valid use: anhedonia/apathy mechanism hypothesis. Invalid claim: amount of pleasure or motivation from MRI. Next data: task/rest fMRI,validated scales.
- `psychosocial / social_brain_networks`: DMN,mentalizing network,amygdala,MTL,connectivity. Modality: `fMRI_T1_questionnaires`. Valid use: exploratory loneliness/social support brain correlate. Invalid claim: social wellbeing or relationship quality from MRI. Next data: validated social scales,replication.
- `omics / gut_brain_microbiome_axis`: microbiome diversity,metabolites,inflammation,brain features. Modality: `microbiome_labs_MRI`. Valid use: exploratory systemic aging and inflammation hypothesis. Invalid claim: microbiome state from brain MRI. Next data: stool/metabolomics,labs,replication.
- `skin_face / face_health_embedding`: deep face embeddings,periorbital features,skin texture. Modality: `face_photos`. Valid use: exploratory mortality/treatment-risk or systemic health marker. Invalid claim: face embedding diagnoses hidden disease. Next data: standardized consented photos,outcome validation.
- `musculoskeletal / craniofacial_posture_link`: jaw relation,head posture,cervical alignment,scoliosis labels. Modality: `CBCT_spine_imaging_3D_photo`. Valid use: exploratory association map only. Invalid claim: skull or jaw shape diagnoses scoliosis. Next data: spine imaging,orthodontic data,replication.
- `model_features / foundation_model_embeddings`: BrainFM/NeuroFM embeddings,feature distances,latent clusters. Modality: `foundation_model_outputs`. Valid use: QC/domain-shift and hypothesis generation. Invalid claim: embeddings validate anatomy or diagnose disease. Next data: labeled cohorts,robustness tests.
- `model_features / radiomics_texture`: texture,shape,wavelet,deep radiomics,latent features. Modality: `MRI_or_multimodal_images`. Valid use: candidate features after strict external validation. Invalid claim: radiomics correlation is biological truth. Next data: locked pipeline,external validation,FDR control.

### tier_4_needs_other_modality

- `cardiometabolic / lipids`: LDL,HDL,TG,ApoB,Lp(a). Modality: `blood_labs`. Valid use: direct cardiometabolic/dementia risk covariates. Invalid claim: cholesterol from T1 brain MRI. Next data: blood panel.
- `cardiometabolic / glucose_insulin`: HbA1c,fasting glucose,insulin,diabetes status. Modality: `blood_labs_EHR`. Valid use: metabolic risk profile. Invalid claim: diabetes status from MRI alone. Next data: labs,EHR,medications.
- `cardiometabolic / kidney_liver_axis`: eGFR,creatinine,ALT/AST,GGT,liver fat. Modality: `labs_body_MRI`. Valid use: systemic aging and vascular covariates. Invalid claim: kidney or liver age from brain T1. Next data: labs,abdominal/body MRI.
- `inflammation_immune / inflammatory_markers`: CRP,IL6,TNF,CBC,NLR. Modality: `blood_labs`. Valid use: inflammaging/allostatic load covariates. Invalid claim: immune age from structural MRI. Next data: blood markers,cell counts.
- `omics / proteomic_organ_age`: brain,heart,artery,kidney,liver immune proteomic clocks. Modality: `plasma_proteomics`. Valid use: organ-specific aging/risk architecture. Invalid claim: organ proteomic age from brain MRI alone. Next data: proteomics,outcome labels.
- `omics / epigenetic_age`: DNAm clocks,pace of aging,telomere length. Modality: `blood_or_tissue_omics`. Valid use: biological age comparator. Invalid claim: methylation age from MRI. Next data: methylation/telomere assay.
- `omics / genetic_risk`: APOE,PRS,rare variants,monogenic disease. Modality: `genotyping_WGS`. Valid use: risk stratification and interaction testing. Invalid claim: genotype from brain MRI. Next data: genotyping,consent,ethics.

### tier_5_stop_invalid

- `impossible_from_brain_t1 / appendicitis`: appendix inflammation,abdominal pain cause. Modality: `abdominal_US_CT_labs`. Valid use: only valid with abdominal/clinical data. Invalid claim: acute appendicitis from structural brain MRI. Next data: abdominal imaging,CBC,clinical exam.
- `impossible_from_brain_t1 / genetic_lung_disease`: CF,alpha-1 antitrypsin deficiency,IPF monogenic risk. Modality: `genetics_pulmonary_tests`. Valid use: only valid with genetics/pulmonary data. Invalid claim: genetic lung disease from brain MRI. Next data: WGS/targeted genetics,PFT,chest imaging.
- `impossible_from_brain_t1 / cholesterol_level`: LDL,ApoB,Lp(a). Modality: `blood_labs`. Valid use: direct lab feature if measured. Invalid claim: lipid panel from T1 MRI. Next data: blood lipids.
- `impossible_from_brain_t1 / immune_age_without_labs`: immune cell aging,cytokines,immunosenescence. Modality: `blood_omics`. Valid use: direct immune profile if measured. Invalid claim: immune age from structural MRI. Next data: CBC,flow cytometry,proteomics.
- `impossible_from_brain_t1 / periodontitis_without_dental_data`: gum inflammation,pocket depth,alveolar bone loss. Modality: `dental_exam_Xray`. Valid use: oral covariate if measured. Invalid claim: gum inflammation from brain MRI. Next data: dental exam.
- `impossible_from_brain_t1 / loneliness_without_behavior_or_self_report`: loneliness,social isolation,support. Modality: `questionnaire_behavior`. Valid use: psychosocial feature if validated. Invalid claim: loneliness from MRI or GPS alone. Next data: validated scales,behavioral data.
- `impossible_from_brain_t1 / true_intracranial_pressure`: ICP in mmHg. Modality: `clinical_measurement`. Valid use: clinical pressure if measured. Invalid claim: true ICP from T1 features. Next data: clinical exam,ophthalmology,LP if indicated.
- `impossible_from_brain_t1 / mirror_neuron_social_wellbeing`: mirror neuron count,social success. Modality: `none`. Valid use: not a valid measurable target. Invalid claim: social wellbeing as mirror neuron quantity. Next data: replace with validated social scales.
- `impossible_from_brain_t1 / remote_organ_cancer_screening`: colon,lung,ovary,pancreas cancer outside field. Modality: `organ_specific_screening`. Valid use: only valid with relevant screening data. Invalid claim: cancer outside field from brain MRI. Next data: appropriate organ imaging/labs.
- `impossible_from_brain_t1 / acute_infection_outside_head`: UTI,pneumonia,appendicitis,sepsis source. Modality: `clinical_labs_imaging`. Valid use: only valid with clinical/lab data. Invalid claim: infection source from brain MRI. Next data: vitals,CBC,CRP,cultures,organ imaging.

## High-Level Coverage

The map covers these families:

- brain structure and atrophy;
- vascular brain injury and perfusion;
- white-matter microstructure and myelin/water/iron;
- neurodegeneration patterns;
- sensory reserve;
- pain, stress, reward, psychiatric and cognitive axes;
- sex, hormones, reproductive transition, and sex-as-moderator;
- cardiometabolic, inflammatory, immune, proteomic, epigenetic, and genetic
  markers;
- retina/OCT/fundus and oculomics;
- dental, periodontal, sinus, skin, face, skull, posture, and musculoskeletal
  features;
- body MRI and organ-specific aging;
- wearables, mobility, loneliness, environment, and exposome;
- explicit impossible or nonsensical targets.

## Core Interpretation

The useful endpoint is not "one brain age number". The useful endpoint is a
profile of measurable axes:

```text
structural + vascular + white_matter + sensory + metabolic + inflammatory
+ psychosocial + mobility_environment + organ_systems + robustness_uncertainty
```

Chronological age remains a calibration variable and confounder. It should not
be the only proxy for biology.

## Boundary Examples

Meaningful:

- WMH burden from FLAIR as a vascular brain-injury marker.
- Microbleeds from SWI/T2* as a small-vessel disease marker.
- Retinal vessel caliber and OCT layer thickness as oculomics features.
- Blood LDL/ApoB/HbA1c/BP as direct cardiometabolic features.
- GPS life-space entropy as a behavioral/environmental feature that must be
  validated against loneliness/social-support questionnaires.

Not meaningful from structural brain MRI alone:

- acute appendicitis;
- genetic lung disease status;
- cholesterol level;
- immune age;
- gum inflammation;
- loneliness;
- exact chronic pain state;
- true intracranial pressure;
- social wellbeing or "mirror neuron count".

These may become meaningful only if the correct modality and labels are added:
abdominal imaging/labs for appendicitis, genomics/pulmonary testing for lung
genetic disease, blood labs for cholesterol and immune age, dental exam for gum
inflammation, questionnaires and behavior data for loneliness.

## Recommended Modeling Shape

For population cohorts:

```text
outcome ~ age_spline + sex + feature + scanner_site + preprocessing + TIV
        + sex:feature + age_spline:feature + socioeconomic_covariates
```

For travelling subjects/test-retest:

```text
feature_or_prediction ~ fixed_covariates + (1 | subject) + (1 | site/scanner)
```

For n=1:

```text
report = feature_value + provenance + QC + uncertainty + "no validation claim"
```

## Sources To Maintain

- STRIVE neuroimaging small-vessel disease standards:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3714437/
- STRIVE-2 update summary:
  https://clinical-brain-sciences.ed.ac.uk/row-fogo-centre-research-ageing-and-brain/news-and-events/news/updated-strive-guidelines-published
- White-matter microstructure in UK Biobank:
  https://www.nature.com/articles/ncomms13629
- MRtrix fixel-based analysis:
  https://mrtrix.readthedocs.io/en/dev/fixel_based_analysis/mt_fibre_density_cross-section.html
- Myelin water fraction review:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11951035/
- Retinal/OCT age and mortality:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10828229/
- Organ aging signatures in plasma proteome:
  https://www.nature.com/articles/s41586-023-06802-1
- Wearable aging clock:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/
- Loneliness neurobiology systematic review:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8258736/
- Chronic pain imaging biomarkers:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8763372/
- Dental age estimation with CBCT:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10315230/
- Periodontitis and dementia/systemic inflammation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11266257/
- Sinus incidental findings:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3636478/
- Body composition profiling in UK Biobank MRI:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6220857/

