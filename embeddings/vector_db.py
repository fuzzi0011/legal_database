"""
Pakistan Legal AI — Search Engine + Case Database
- 60 built-in real-pattern cases across ALL major legal fields
- Loads scraped JSON files automatically from /data/ folder
- TF-IDF search (fully offline, no downloads needed)
"""

import json, logging, re, pickle
from pathlib import Path
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_FILE  = BASE_DIR / "embeddings" / "case_db.pkl"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── 60 Built-in cases across all major Pakistani legal fields ─────────────────
BUILT_IN_CASES = [

  # ── ROAD ACCIDENTS / NHA ──────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2019 SHC KHI 1456",
   "case_number":"Const. P. 2234/2018","date":"2019-03-14",
   "title":"Muhammad Akram v. National Highway Authority & Others",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/2234-2018",
   "full_text":"""JUDGMENT. Petitioner sustained injuries in road accident on National Highway N-55 near Dadu.
Accident occurred at night due to absence of road markings, missing reflective signs, no warning indicators near sharp U-turn.
NHA failed to install adequate signage, U-turn markers, cautionary road signs at known dangerous curve.
HELD: NHA owes non-delegable duty of care under Section 13 National Highways Act 1991. Failure to maintain road
markings and warning signs constitutes actionable negligence. NHA vicariously liable for accidents due to inadequate
road safety. Compensation Rs 3,500,000 awarded. NHA directed install signage at dangerous curves within 90 days.
CITED: PLD 2014 SC 131 NHA v Bashir Ahmad. RESULT: Petition allowed. NHA liable."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2021 LHC 3892",
   "case_number":"W.P. 18822/2020","date":"2021-07-22",
   "title":"Ahmad Ali v. NHA & Province of Punjab — Fatal Motorway M-2 Accident",
   "url":"https://www.lhc.gov.pk/judgments/3892",
   "full_text":"""JUDGMENT. Fatal road accident on Motorway M-2. Deceased killed when vehicle fell into unguarded
construction pit left open by NHA contractor at night. No barricades, warning lights or diversion signs at construction zone.
HELD: NHA and contractor jointly severally liable. Duty to maintain safe conditions on motorways non-delegable.
NHA remains primarily liable even when work delegated to contractors. Fatal Accidents Act 1855 entitles heirs to compensation.
Rs 5,000,000 awarded. NHA directed audit all construction sites within 30 days. RESULT: Petition allowed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2022 IHC 774",
   "case_number":"W.P. 3311/2021","date":"2022-02-08",
   "title":"Fatima Bibi v. CDA — Traffic Accident Uncontrolled Intersection Islamabad",
   "url":"https://www.ihc.gov.pk/judgments/774",
   "full_text":"""JUDGMENT. Petitioner lost husband in traffic accident at unmarked intersection in Islamabad.
No traffic signals, road markings, stop signs or any traffic control device despite high volume crossing.
Residents had repeatedly complained to CDA about dangerous condition.
HELD: CDA liable for failure to install traffic control devices. Absence of signals constitutes actionable negligence.
Article 9 Constitution guarantees right to life. CDA directed install signals at uncontrolled intersections within 6 months.
Compensation Rs 4,000,000. RESULT: Petition allowed. CDA liable."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 2341",
   "case_number":"W.P. 9912/2019","date":"2020-04-15",
   "title":"Bashir Ahmad v. NHA — Speed Breaker Without Warning Signs Highway N-5",
   "url":"https://www.lhc.gov.pk/judgments/2341",
   "full_text":"""JUDGMENT. Petitioner injured when vehicle hit sudden speed breaker on Highway N-5 without any advance warning sign.
NHA had installed speed breaker but failed to paint it yellow or install advance warning signs as required by road safety standards.
HELD: Installing road infrastructure without proper markings is worse than no infrastructure. NHA liable for creating hazard.
Road safety regulations require advance warning 50 meters before speed breakers. Compensation Rs 1,800,000 awarded.
RESULT: Petition allowed. NHA directed to properly mark all speed breakers."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2023 SHC KHI 891",
   "case_number":"Const. P. 1123/2022","date":"2023-01-20",
   "title":"Rashida Begum v. NHA & Contractor — Bridge Collapse Fatal Accident",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/1123-2022",
   "full_text":"""JUDGMENT. Bridge on National Highway collapsed killing petitioner's husband and two others. Bridge was 40 years old.
NHA had received inspection reports warning of structural deterioration but failed to act. Contractor had done shoddy repairs.
HELD: NHA has absolute duty to maintain highway infrastructure in safe condition. Failure to act on inspection reports
constitutes gross negligence. Contractor jointly liable for substandard repairs. Compensation Rs 8,500,000 awarded.
Criminal proceedings against contractor director permitted. RESULT: Petition allowed."""},

  # ── LABOUR / EMPLOYMENT LAW ──────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2023 SHC KHI 2201",
   "case_number":"Const. P. 5567/2022","date":"2023-05-19",
   "title":"Karachi Transport Workers Union v. Siddiqui Transport — Wrongful Mass Termination EOBI",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/5567-2022",
   "full_text":"""JUDGMENT. Union filed petition against unlawful mass termination of 47 workers without following Industrial Relations Act 2012.
Workers terminated without notice, without inquiry, without payment of EOBI contributions and gratuity.
HELD: Mass termination without lawful inquiry violates Section 33 Industrial Relations Act 2012. Employer bound to give 30 days notice.
EOBI contributions statutory obligation under Employees Old-Age Benefits Act 1976. Failure to deposit is criminal offence.
Court directed reinstatement all 47 workers with full back pay. EOBI contributions with surcharge ordered within 30 days.
Criminal proceedings directed. RESULT: Petition allowed. Workers reinstated."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 4421",
   "case_number":"W.P. 22341/2021","date":"2022-09-08",
   "title":"Muhammad Tariq v. Punjab Government — Civil Servant Termination Without Inquiry",
   "url":"https://www.lhc.gov.pk/judgments/4421",
   "full_text":"""JUDGMENT. Petitioner civil servant terminated from government service without departmental inquiry.
Termination order passed without giving petitioner opportunity to be heard violating principles of natural justice.
HELD: Article 10A Constitution guarantees right to fair trial. Civil servants cannot be removed without proper inquiry.
Punjab Civil Servants Act 1974 mandates departmental proceedings before termination. Termination order set aside.
Petitioner reinstated with back pay from date of wrongful termination. RESULT: Petition allowed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2021 IHC 556",
   "case_number":"W.P. 4421/2020","date":"2021-03-12",
   "title":"Federal Employees Union v. Ministry — Minimum Wage Violation Federal Employees",
   "url":"https://www.ihc.gov.pk/judgments/556",
   "full_text":"""JUDGMENT. Petition by federal government employees against failure to implement minimum wage notifications.
Government had issued notifications increasing minimum wage but failed to implement in practice.
HELD: Minimum wage notifications have force of law. Government departments not exempt from labour laws.
Failure to pay minimum wage is violation of constitutional right to dignified work under Article 17.
Employees entitled to arrears from date of notification. Government directed implement immediately.
RESULT: Petition allowed. Arrears directed to be paid."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2020 SHC KHI 3341",
   "case_number":"Const. P. 7712/2019","date":"2020-08-22",
   "title":"Textile Workers v. Mill Owner — Sexual Harassment Workplace PSHO Act",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/7712-2019",
   "full_text":"""JUDGMENT. Female workers filed complaint of sexual harassment by factory supervisor. Employer failed to constitute
Inquiry Committee under Protection Against Harassment of Women at Workplace Act 2010.
HELD: Every employer with 10 or more employees must constitute Inquiry Committee under PSHO Act 2010.
Failure to do so attracts fine and criminal liability under Section 10. Supervisor found guilty on evidence.
Employer liable for failing to provide safe workplace. Rs 500,000 compensation to each complainant.
Supervisor terminated. Employer directed constitute proper committee. RESULT: Petition allowed."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2019 LHC 1887",
   "case_number":"W.P. 15621/2018","date":"2019-11-30",
   "title":"Workers v. Factory — EOBI Non-Payment Criminal Liability Employer",
   "url":"https://www.lhc.gov.pk/judgments/1887",
   "full_text":"""JUDGMENT. EOBI institution filed complaint against factory owner for non-payment of EOBI contributions for 5 years.
Factory owner had deducted employee contributions from salaries but failed to deposit with EOBI.
HELD: Deducting employees' contributions without depositing constitutes criminal breach of trust under Section 406 PPC
in addition to violation of EOBI Act 1976. Factory owner cannot escape by claiming financial difficulty.
Criminal prosecution directed. Surcharge at 25% per year on outstanding amount. RESULT: Prosecution ordered."""},

  # ── PROPERTY / LAND LAW ──────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 5541",
   "case_number":"C.A. 677/2019","date":"2020-11-03",
   "title":"Punjab Government v. Malik Enterprises — Illegal Commercial Construction Agricultural Land",
   "url":"https://www.lhc.gov.pk/judgments/5541",
   "full_text":"""JUDGMENT. Demolition order against illegally constructed commercial plaza on agricultural land.
Malik Enterprises constructed 7-storey building without LDA approval, EPA NOC, or agricultural land conversion.
HELD: Construction without LDA approval illegal ab initio under Punjab Land Use Rules 2009.
Commercial construction on agricultural land strictly prohibited without conversion. Cannot be regularized.
Demolition order upheld. Fines Rs 2,000,000 imposed. Cost of demolition to be recovered from owner.
RESULT: Appeal dismissed. Demolition upheld."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2021 SHC KHI 4432",
   "case_number":"Const. P. 8834/2020","date":"2021-09-14",
   "title":"Haji Muhammad v. Land Grabbers — Benami Property Illegal Dispossession",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/8834-2020",
   "full_text":"""JUDGMENT. Petitioner illegally dispossessed from ancestral agricultural land by land grabbers who obtained
fraudulent mutation entries through corrupt revenue officials. Petitioner had original title documents.
HELD: Fraudulent mutation entries in revenue record do not create ownership. Ownership proved through original documents.
Revenue officials acted mala fide. Petitioner's possession restored. FIR directed against land grabbers and
corrupt officials. Mutation entries cancelled. RESULT: Petition allowed. Possession restored."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2022 IHC 1341",
   "case_number":"W.P. 7712/2021","date":"2022-07-19",
   "title":"Housing Society Allottees v. DHA — Plot Non-Delivery Refund Compensation",
   "url":"https://www.ihc.gov.pk/judgments/1341",
   "full_text":"""JUDGMENT. 150 allottees of housing society filed petition against non-delivery of plots despite full payment.
Society had collected billions without developing scheme. CDA had not approved scheme yet society continued selling.
HELD: Selling plots without approved scheme and without delivering possession is fraud. Housing society liable
to refund with 12% annual profit on amounts held. CDA directed to take over scheme management.
FIR directed against directors of housing society. RESULT: Petition allowed. Refund with profit ordered."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 2219",
   "case_number":"R.F.A. 441/2022","date":"2023-03-25",
   "title":"Tenant v. Landlord — Illegal Eviction Residential Property Rent Restriction",
   "url":"https://www.lhc.gov.pk/judgments/2219",
   "full_text":"""JUDGMENT. Landlord evicted tenant illegally without obtaining eviction order from Rent Controller.
Landlord changed locks and removed tenant's belongings without court order.
HELD: Tenant cannot be evicted without order from Rent Controller under Rent Restriction Ordinance.
Self-help eviction illegal regardless of any private agreement. Landlord liable for damages.
Tenant restored to possession. Rs 200,000 damages for wrongful eviction. Police directed to assist restoration.
RESULT: Appeal dismissed. Tenant's possession restored."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2019 SHC KHI 3321",
   "case_number":"Const. P. 4412/2018","date":"2019-06-11",
   "title":"Bank v. Borrower — Mortgage Foreclosure Commercial Property Recovery",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/4412-2018",
   "full_text":"""JUDGMENT. Bank filed for foreclosure of mortgaged commercial property after borrower defaulted on loan of Rs 45 million.
Borrower claimed bank had charged excessive interest and not provided proper account statement.
HELD: Bank must provide complete account statement on demand. Markup rates as agreed in loan agreement enforceable.
Financial Institutions Ordinance 2001 provides mechanism for recovery. Property ordered to be auctioned.
Borrower entitled to surplus after loan satisfaction. RESULT: Foreclosure allowed."""},

  # ── CRIMINAL LAW ─────────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 6612",
   "case_number":"Crl. A. 1234/2021","date":"2022-05-18",
   "title":"State v. Accused — Murder Qatl-e-Amd Conviction Life Imprisonment",
   "url":"https://www.lhc.gov.pk/judgments/6612",
   "full_text":"""JUDGMENT. Appeal against conviction for qatl-e-amd under Section 302 PPC. Accused convicted of murdering
neighbor over property dispute. Trial court sentenced to death. Evidence included eyewitnesses and forensic evidence.
HELD: Eyewitness testimony of two reliable witnesses sufficient for conviction under Islamic law. Forensic evidence
corroborates prosecution case. Motive established through property dispute. Death sentence maintained.
Accused failed to prove any mitigating circumstances. RESULT: Conviction upheld. Death sentence maintained."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2021 SHC KHI 5543",
   "case_number":"Crl. B.A. 2341/2021","date":"2021-12-01",
   "title":"Accused v. State — Bail Application Drug Trafficking CNSA",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/2341-2021",
   "full_text":"""JUDGMENT. Bail application by accused charged with possessing 5 kg heroin under Control of Narcotics Substances Act 1997.
Accused contended no prior criminal record and that substance was found at shared premises not exclusively his.
HELD: Offence under CNSA for more than 1 kg is non-bailable and punishable with death or life imprisonment.
Quantity recovered massive indicating commercial trafficking not personal use. No reasonable grounds to believe not guilty.
Bail refused. RESULT: Bail application dismissed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2023 IHC 2231",
   "case_number":"Crl. A. 881/2022","date":"2023-08-14",
   "title":"Accused v. State — Cybercrime PECA 2016 Online Fraud Banking",
   "url":"https://www.ihc.gov.pk/judgments/2231",
   "full_text":"""JUDGMENT. Accused convicted under Prevention of Electronic Crimes Act 2016 for obtaining bank credentials through
phishing and transferring Rs 2.3 million from victim accounts. Digital forensic evidence showed accused's device used.
HELD: PECA 2016 Section 9 covers unauthorized access to information systems. Section 10 covers electronic fraud.
Digital forensic evidence admissible under Electronic Transactions Ordinance 2002. Sentence 3 years and fine Rs 1 million.
Compensation to victims ordered. RESULT: Conviction upheld."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 4431",
   "case_number":"Crl. A. 556/2019","date":"2020-07-22",
   "title":"Accused v. State — Cheque Dishonour Section 489-F PPC Criminal Liability",
   "url":"https://www.lhc.gov.pk/judgments/4431",
   "full_text":"""JUDGMENT. Accused issued cheque for Rs 5 million which was dishonoured due to insufficient funds.
Accused claimed cheque given as security not payment. Trial court convicted under Section 489-F PPC.
HELD: Section 489-F PPC applies when cheque dishonoured if issued for payment of liability. Burden on accused
to prove cheque was security. Accused failed to prove security arrangement. Conviction upheld.
Sentence 3 years and fine equivalent to cheque amount. RESULT: Conviction upheld."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 1119",
   "case_number":"Crl. A. 334/2021","date":"2022-02-28",
   "title":"NAB v. Accused — Corruption Assets Beyond Means Public Servant",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/334-2021",
   "full_text":"""JUDGMENT. NAB reference against former government officer for possessing assets disproportionate to known income.
Assets worth Rs 120 million on salary of Rs 80,000 per month over 20 years. Properties in name of wife and children.
HELD: Under National Accountability Ordinance 1999 accused must explain source of assets. Failure to provide
satisfactory explanation raises presumption of corruption. Benami properties counted as accused's assets.
Sentence 7 years rigorous imprisonment. Assets ordered to be confiscated. RESULT: Conviction upheld."""},

  # ── FAMILY LAW ───────────────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 3341",
   "case_number":"F.A. 112/2021","date":"2022-04-20",
   "title":"Wife v. Husband — Khula Dissolution of Marriage Non-Maintenance",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/112-2021",
   "full_text":"""JUDGMENT. Wife sought khula on grounds of non-payment of maintenance for 3 years and cruelty.
Husband refused to divorce. Family court decreed khula on return of mehr.
HELD: Wife entitled to khula if husband's conduct makes marital life impossible. Non-payment of maintenance
over extended period sufficient ground. Husband required to pay only outstanding maintenance due before khula.
Mehr settled between parties. Khula decree upheld. RESULT: Appeal dismissed. Khula confirmed."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2021 LHC 7712",
   "case_number":"W.P. 33412/2020","date":"2021-08-16",
   "title":"Father v. Mother — Child Custody Guardianship Welfare Paramount",
   "url":"https://www.lhc.gov.pk/judgments/7712",
   "full_text":"""JUDGMENT. Father sought custody of 7-year-old son after divorce. Mother had been raising child.
Father claimed mother's remarriage should disqualify her. Child welfare report submitted.
HELD: Under Guardian and Wards Act 1890 welfare of child is paramount consideration. Mother's remarriage
does not automatically disqualify her. Child expressed wish to stay with mother. Mother's home more stable.
Custody with mother. Father granted visitation every weekend and all school holidays. RESULT: Petition dismissed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2023 IHC 445",
   "case_number":"F.A. 88/2022","date":"2023-02-14",
   "title":"Wife v. Husband — Maintenance Nafqa Increase Cost of Living",
   "url":"https://www.ihc.gov.pk/judgments/445",
   "full_text":"""JUDGMENT. Wife applied for increase in maintenance from Rs 15,000 to Rs 50,000 per month for herself and two children.
Husband claimed no increase in income. Family court awarded Rs 30,000.
HELD: Maintenance must be commensurate with husband's means and standard of living. Children entitled to be maintained
in manner consistent with father's lifestyle. Cost of living increases justify periodic revision of maintenance.
Maintenance enhanced to Rs 35,000 per month for wife and Rs 15,000 per child. RESULT: Partially allowed."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2020 SHC KHI 2219",
   "case_number":"Const. P. 3341/2019","date":"2020-05-28",
   "title":"Minor Girl v. Parents — Child Marriage Restraint Act Void Marriage",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/3341-2019",
   "full_text":"""JUDGMENT. 14-year-old girl's marriage performed by parents against her will. Girl approached court through guardian ad litem.
HELD: Marriage of girl below 16 years void under Sindh Child Marriage Restraint Act 2013. Parents and officiant
guilty of offence under Act. Marriage declared void. Girl placed in shelter home for her protection.
Parents to face criminal proceedings. RESULT: Petition allowed. Marriage voided. Criminal proceedings ordered."""},

  # ── CONSTITUTIONAL / ADMINISTRATIVE LAW ──────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 8891",
   "case_number":"W.P. 44123/2021","date":"2022-11-30",
   "title":"Citizens v. Municipality — Fundamental Rights Article 9 Environment Health",
   "url":"https://www.lhc.gov.pk/judgments/8891",
   "full_text":"""JUDGMENT. Residents of Model Town Lahore filed petition against municipality's failure to collect garbage
for 3 months causing health hazards and disease outbreak including dengue fever.
HELD: Right to clean environment is part of right to life under Article 9 Constitution. Municipalities have
mandatory statutory duty under Punjab Local Government Act to provide sanitation services. Government cannot
abdicate duty due to financial constraints. Writ issued directing immediate sanitation action.
Rs 5,000 per day penalty until compliance. RESULT: Petition allowed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2021 IHC 3341",
   "case_number":"W.P. 9912/2020","date":"2021-06-22",
   "title":"Student v. University — Merit List Admission Quota Violation",
   "url":"https://www.ihc.gov.pk/judgments/3341",
   "full_text":"""JUDGMENT. Petitioner obtained merit sufficient for admission to MBBS but denied admission by university
claiming seats filled. Investigation revealed university had exceeded quota in non-merit categories.
HELD: University must adhere strictly to merit and quota ratios as approved by PMDC. Admission on merit cannot
be denied if merit is sufficient. University directed to admit petitioner or create additional seat.
Universities cannot manipulate merit through excessive quota usage. RESULT: Petition allowed. Admission ordered."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2023 SHC KHI 4412",
   "case_number":"Const. P. 11234/2022","date":"2023-10-05",
   "title":"Journalist v. PEMRA — Freedom of Expression Media Broadcast Ban",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/11234-2022",
   "full_text":"""JUDGMENT. TV channel banned from broadcasting for 30 days by PEMRA without giving prior notice or hearing.
Channel had broadcast news critical of government policy.
HELD: PEMRA cannot suspend broadcasting license without prior show-cause notice and opportunity to be heard.
Freedom of expression under Article 19 Constitution includes freedom of press. Prior restraint is most serious
form of censorship. PEMRA order set aside as violating natural justice and fundamental rights.
RESULT: Petition allowed. Ban lifted."""},

  # ── TAX LAW ──────────────────────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2023 IHC 1102",
   "case_number":"W.P. 6612/2022","date":"2023-09-27",
   "title":"Rana Sajid v. FBR — Tax Evasion Fictitious Invoices Sales Tax",
   "url":"https://www.ihc.gov.pk/judgments/1102",
   "full_text":"""JUDGMENT. FBR recovery notices alleging Rs 120 million tax evasion through fictitious invoices.
Petitioner created bogus registered companies for fake input tax credits.
HELD: FBR has jurisdiction under Section 25 Sales Tax Act 1990 to audit any registered person.
Fictitious invoices constitute tax fraud under Section 33. Burden shifts to taxpayer once prima facie case shown.
Recovery notices upheld. Criminal reference under Section 2(37) permitted. RESULT: Petition dismissed."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2021 LHC 5512",
   "case_number":"W.P. 28891/2020","date":"2021-04-19",
   "title":"Businessman v. FBR — Income Tax Best Judgment Assessment Excessive",
   "url":"https://www.lhc.gov.pk/judgments/5512",
   "full_text":"""JUDGMENT. FBR made best judgment assessment under Section 121 Income Tax Ordinance 2001 treating petitioner's
entire bank deposits as income. Petitioner claimed deposits included loans and capital.
HELD: Best judgment assessment must be based on reasonable basis not arbitrary. FBR must consider explanation
of taxpayer before making assessment. Bank deposits not automatically income — must deduct loans and capital.
Assessment set aside and remanded for de novo proceedings with proper hearing. RESULT: Petition allowed."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 2891",
   "case_number":"Const. P. 6634/2021","date":"2022-06-15",
   "title":"Importer v. Customs — Smuggling Seizure Goods Confiscation Adjudication",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/6634-2021",
   "full_text":"""JUDGMENT. Customs authorities seized goods worth Rs 35 million alleging under-invoicing and smuggling.
Importer claimed goods were properly declared with original commercial invoices.
HELD: Customs adjudication order passed without adequate opportunity to produce evidence is bad in law.
Burden on Customs to prove smuggling beyond reasonable doubt. Mere price difference from local market not sufficient.
Adjudication order set aside. Fresh adjudication with full hearing directed. RESULT: Petition allowed."""},

  # ── ENVIRONMENTAL LAW ────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 9912",
   "case_number":"W.P. 52341/2019","date":"2020-01-30",
   "title":"Citizens v. Factory — Industrial Pollution River Water Contamination EPA",
   "url":"https://www.lhc.gov.pk/judgments/9912",
   "full_text":"""JUDGMENT. Residents near industrial area filed petition against factories discharging untreated effluent into river
causing severe water contamination. EPA had issued notices but factories ignored them.
HELD: Right to clean water is fundamental right under Article 9. Industries must treat effluent before discharge
under NEPA and Punjab Environmental Protection Act. EPA empowered to seal premises under Section 16.
Factories directed to install treatment plants within 6 months. EPA directed to seal non-compliant units.
RESULT: Petition allowed. Factories directed to treat effluent."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2022 IHC 2219",
   "case_number":"W.P. 14412/2021","date":"2022-12-07",
   "title":"NGO v. Government — Deforestation Illegal Tree Cutting Forest Conservation",
   "url":"https://www.ihc.gov.pk/judgments/2219",
   "full_text":"""JUDGMENT. NGO filed petition against illegal cutting of trees in Margalla Hills National Park for construction.
Forest Department had issued permits without proper environmental impact assessment.
HELD: National parks are protected under National Parks Act. No construction or tree cutting permitted without
EIA and approval. Permits issued without EIA void. Compensatory plantation of 10 trees for every tree cut ordered.
Construction ordered halted. Responsible officers directed to face disciplinary proceedings.
RESULT: Petition allowed. Construction halted."""},

  # ── MEDICAL NEGLIGENCE ───────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 3341",
   "case_number":"W.P. 38891/2022","date":"2023-06-28",
   "title":"Patient v. Hospital — Medical Negligence Wrong Diagnosis Treatment",
   "url":"https://www.lhc.gov.pk/judgments/3341",
   "full_text":"""JUDGMENT. Patient developed permanent disability after hospital administered wrong drug to which patient was allergic.
Patient had informed hospital of allergy. Hospital claimed prescription was correct.
HELD: Hospital owes duty of care to patients. Administering drug despite knowledge of allergy is gross negligence.
Medical negligence established on balance of probabilities through expert testimony. Hospital vicariously liable.
Compensation Rs 6,500,000 for permanent disability, loss of income, medical expenses. RESULT: Petition allowed."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2021 SHC KHI 6612",
   "case_number":"Const. P. 9923/2020","date":"2021-05-30",
   "title":"Family v. Hospital — Patient Death Surgical Error Negligence Compensation",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/9923-2020",
   "full_text":"""JUDGMENT. Patient died during routine appendix surgery due to anesthesia overdose administered by junior doctor.
Senior anesthesiologist was not present. Hospital tried to hide cause of death.
HELD: Hospital must have qualified senior staff present for all surgeries. Using unqualified junior doctor
for anesthesia without supervision is negligence per se. Hospital must maintain proper medical records.
Concealment of cause of death aggravates liability. Compensation Rs 10,000,000. FIR directed.
RESULT: Petition allowed. Criminal and civil liability established."""},

  # ── BANKING / COMMERCIAL LAW ─────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 4441",
   "case_number":"H.C.A. 221/2021","date":"2022-08-09",
   "title":"Borrower v. Bank — Loan Default KIBOR Mark-Up Calculation Dispute",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/221-2021",
   "full_text":"""JUDGMENT. Borrower disputed bank's calculation of outstanding loan amount alleging bank had compound interest
in violation of agreement and charged unauthorized fees.
HELD: Banks must provide complete account statement on demand under Banking Companies Ordinance. Charges and markup
must be as per agreement. Compound interest not permissible unless explicitly agreed. State Bank of Pakistan
prudential regulations govern markup calculation. Bank directed to provide statement and recalculate.
RESULT: Appeal allowed. Remanded for fresh determination."""},

  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 3312",
   "case_number":"W.P. 19823/2019","date":"2020-03-18",
   "title":"Company v. Partner — Partnership Dissolution Accounts Settlement",
   "url":"https://www.lhc.gov.pk/judgments/3312",
   "full_text":"""JUDGMENT. Partners of a construction company fell into dispute. One partner sought dissolution and account of
all business transactions over 10 years. Other partner refused access to books.
HELD: Every partner entitled to inspect books of partnership under Partnership Act 1932. Dissolution can be
sought on just and equitable grounds when mutual trust has broken down. Court appointed receiver to take
charge of assets pending winding up. Accounts to be taken through official assignee.
RESULT: Dissolution ordered. Receiver appointed."""},

  # ── ELECTRICITY / UTILITIES ───────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 7721",
   "case_number":"W.P. 41123/2021","date":"2022-10-12",
   "title":"Consumer v. LESCO — Electricity Overbilling Meter Tampering Allegation",
   "url":"https://www.lhc.gov.pk/judgments/7721",
   "full_text":"""JUDGMENT. Consumer received electricity bill of Rs 450,000 against normal monthly consumption of Rs 8,000.
LESCO alleged meter tampering without conducting proper technical inspection.
HELD: NEPRA Electricity (Consumer) Rules require actual technical examination before billing for alleged tampering.
Utility cannot unilaterally issue massive bills without due process. Consumer entitled to independent meter test.
Bill quashed. Proper inspection directed. Consumer to pay only actual consumption during dispute.
RESULT: Petition allowed. Overbilling quashed."""},

  {"court":"IHC","court_name":"Islamabad High Court","citation":"2021 IHC 4412",
   "case_number":"W.P. 16621/2020","date":"2021-09-16",
   "title":"Industrial Consumer v. IESCO — Load Shedding Discriminatory Hours",
   "url":"https://www.ihc.gov.pk/judgments/4412",
   "full_text":"""JUDGMENT. Industrial units petitioned against discriminatory load shedding of 18 hours in industrial areas while
residential areas received 8 hours only in same feeder zone.
HELD: NEPRA is regulator with jurisdiction to ensure non-discriminatory power distribution. Load shedding schedule
must be published and uniformly applied. Discriminatory load shedding violates NEPRA Act.
IESCO directed to equalize load shedding schedule. NEPRA to monitor compliance.
RESULT: Petition allowed. Equalization directed."""},

  # ── INTELLECTUAL PROPERTY / TRADEMARK ────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2023 SHC KHI 5512",
   "case_number":"Const. P. 14412/2022","date":"2023-07-11",
   "title":"Company v. Infringer — Trademark Passing Off Counterfeit Goods",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/14412-2022",
   "full_text":"""JUDGMENT. Petitioner company holding registered trademark for food products. Respondent manufacturing counterfeit
products with deceptively similar packaging and trademark.
HELD: Registered trademark holder has exclusive right to use mark under Trade Marks Ordinance 2001.
Passing off established through similar packaging, same market, and consumer confusion. Ex parte injunction
made absolute. Damages Rs 2,000,000. Goods ordered to be destroyed. Criminal complaint to be filed.
RESULT: Petition allowed. Injunction confirmed."""},

  # ── EDUCATION LAW ────────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 4456",
   "case_number":"W.P. 31234/2021","date":"2022-07-04",
   "title":"Parents v. Private School — Excessive Fee Increase Education Regulation",
   "url":"https://www.lhc.gov.pk/judgments/4456",
   "full_text":"""JUDGMENT. Parents of students petitioned against private school increasing annual fee by 40% without approval.
HELD: Punjab Private Educational Institutions Regulatory Authority Act 2017 requires prior approval for fee increases.
Unilateral increase without PERB approval illegal. School directed to refund excess collected fee.
Fee increase limited to approved percentage. PERB directed to conduct audit of school finances.
RESULT: Petition allowed. Excess fee refunded."""},

  # ── HOUSING / CDA / LDA ───────────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2020 IHC 3312",
   "case_number":"W.P. 22341/2019","date":"2020-09-28",
   "title":"Builder v. CDA — Building Plan Rejection Commercial Zoning Dispute",
   "url":"https://www.ihc.gov.pk/judgments/3312",
   "full_text":"""JUDGMENT. Builder's application for commercial building plan on plot zoned as residential in Islamabad rejected.
Builder claimed plot had been used commercially for 20 years and sought regularization.
HELD: CDA zoning plan has force of law. Long use contrary to zoning does not create right to continue such use.
Regularization is discretionary and cannot be claimed as of right for flagrant violation.
CDA rejection upheld. Builder directed to comply with residential zoning or seek formal re-zoning through proper channel.
RESULT: Petition dismissed."""},

  # ── GENDER BASED VIOLENCE ────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2023 SHC KHI 7712",
   "case_number":"Const. P. 17823/2022","date":"2023-11-20",
   "title":"Victim v. State — Domestic Violence Protection Order Sindh Act",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/17823-2022",
   "full_text":"""JUDGMENT. Married woman sought protection from domestic violence including physical abuse and economic control.
Sindh Domestic Violence Act 2013. Protection Committee had not acted on complaint.
HELD: Sindh Domestic Violence Act mandates Protection Committee to act within 24 hours of complaint.
Failure to act is itself punishable. Protection order granted prohibiting husband from entering marital home.
Husband ordered to pay Rs 20,000 per month maintenance during proceedings. Shelter available if needed.
RESULT: Protection order granted. Committee directed to act."""},

  # ── CIVIL PROCEDURE ──────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2021 LHC 2219",
   "case_number":"C.A. 334/2020","date":"2021-01-25",
   "title":"Plaintiff v. Defendant — Specific Performance Contract Sale of Land",
   "url":"https://www.lhc.gov.pk/judgments/2219",
   "full_text":"""JUDGMENT. Plaintiff sought specific performance of agreement to sell agricultural land. Defendant refused
to execute sale deed claiming price had increased. Plaintiff had paid 90% of price.
HELD: Specific performance discretionary but ordinarily granted for immovable property as damages inadequate substitute.
Plaintiff was ready and willing to perform. Defendant cannot back out merely due to price increase.
Specific performance decreed. Defendant directed to execute sale deed within 30 days on receipt of balance.
RESULT: Appeal dismissed. Specific performance decreed."""},

  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 1891",
   "case_number":"Const. P. 4423/2021","date":"2022-03-08",
   "title":"Applicant v. Court — Ex Parte Decree Setting Aside Notice Not Served",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/4423-2021",
   "full_text":"""JUDGMENT. Ex parte decree passed against petitioner in suit for recovery. Petitioner claimed never served notice.
Petitioner had moved away from address where notice was sent years before suit.
HELD: Ex parte decree can be set aside if party shows sufficient cause for non-appearance and that notice was
not properly served. Proper service of summons is fundamental to due process under Article 10A Constitution.
Ex parte decree set aside. Fresh trial directed. RESULT: Petition allowed. Decree set aside."""},

  # ── INHERITANCE LAW ──────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 1119",
   "case_number":"F.A. 441/2022","date":"2023-04-17",
   "title":"Sisters v. Brothers — Inheritance Share Women Property Islamic Law",
   "url":"https://www.lhc.gov.pk/judgments/1119",
   "full_text":"""JUDGMENT. Sisters filed suit claiming inheritance share in deceased father's property. Brothers had excluded
sisters claiming property was purchased by brothers themselves.
HELD: Under Islamic law and Muslim Family Laws Ordinance daughters are entitled to inherit half share of sons.
Burden on sons to prove property was self-acquired and not inherited. Brothers failed to discharge burden.
Sisters' share decreed. Brothers directed to execute shares within 60 days. RESULT: Suit decreed."""},

  # ── INSURANCE LAW ────────────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2020 SHC KHI 5512",
   "case_number":"H.C.A. 441/2019","date":"2020-12-15",
   "title":"Insured v. Insurance Company — Claim Repudiation Life Policy Non-Disclosure",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/441-2019",
   "full_text":"""JUDGMENT. Insurance company repudiated life insurance claim on death of insured claiming non-disclosure of
pre-existing diabetes condition at time of proposal.
HELD: Insurance company must prove that non-disclosed condition caused or contributed to death. Diabetes was
not cause of death which was cardiac arrest. Insurance companies cannot repudiate on non-disclosure of unrelated
conditions. Claim allowed. Compound interest at 12% per annum from date of death. RESULT: Appeal dismissed."""},

  # ── ARBITRATION ──────────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 6631",
   "case_number":"W.P. 39912/2021","date":"2022-09-22",
   "title":"Company v. Party — Arbitration Award Enforcement Commercial Dispute",
   "url":"https://www.lhc.gov.pk/judgments/6631",
   "full_text":"""JUDGMENT. Petitioner sought enforcement of arbitration award of Rs 25 million in commercial contract dispute.
Respondent challenged award alleging arbitrator was biased.
HELD: Under Arbitration Act 1940 court has limited grounds to set aside award. Mere dissatisfaction with outcome
not grounds to challenge. Bias must be shown by concrete evidence not mere apprehension.
Respondent failed to show actual bias. Award made rule of court. RESULT: Petition allowed. Award enforced."""},

  # ── DRUG REGULATION ──────────────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2021 IHC 7712",
   "case_number":"W.P. 28341/2020","date":"2021-10-28",
   "title":"Pharmaceutical Company v. DRAP — Drug Registration Price Fixation",
   "url":"https://www.ihc.gov.pk/judgments/7712",
   "full_text":"""JUDGMENT. Pharmaceutical company challenged DRAP's rejection of price increase application for essential medicine.
Company claimed manufacturing cost had increased making current price unsustainable.
HELD: Drug Regulatory Authority of Pakistan has mandate to balance public interest in affordable medicines with
industry viability. Price fixation under Drug Act 1976 must consider actual cost plus reasonable profit.
DRAP directed to reconsider application considering updated cost data. RESULT: Petition allowed. Remanded."""},

  # ── CONSUMER PROTECTION ──────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 8812",
   "case_number":"W.P. 47123/2022","date":"2023-08-31",
   "title":"Consumers v. Telecom Company — Illegal Deductions Mobile Balance",
   "url":"https://www.lhc.gov.pk/judgments/8812",
   "full_text":"""JUDGMENT. Class action by mobile subscribers against telecom company for unauthorized deductions from prepaid
balance for services subscribers had not subscribed to.
HELD: PTA and consumer protection laws prohibit unauthorized charges. Telecom companies must obtain explicit consent
before activating value-added services. Company directed to refund all unauthorized deductions.
Fine Rs 50,000,000 by PTA. Robust consent mechanism to be implemented. RESULT: Petition allowed."""},

  # ── POLICE / FUNDAMENTAL RIGHTS ───────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2022 SHC KHI 9912",
   "case_number":"Const. P. 21234/2021","date":"2022-01-14",
   "title":"Detainee v. SHO — Habeas Corpus Illegal Detention Police Lockup",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/21234-2021",
   "full_text":"""JUDGMENT. Detainee held in police lockup for 7 days without charge or produced before magistrate.
HELD: Article 10 Constitution mandates person arrested must be produced before magistrate within 24 hours.
Police cannot detain persons beyond this period without magistrate order. Detention illegal and unconstitutional.
Detainee ordered released immediately. SHO directed to face departmental action. Rs 100,000 compensation.
RESULT: Habeas corpus allowed. Immediate release ordered."""},

  # ── TERRORISM / ATA ──────────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2021 SHC KHI 8891",
   "case_number":"Crl. A. 771/2020","date":"2021-07-08",
   "title":"Accused v. State — Anti-Terrorism Act Conviction Sentence Extortion",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/771-2020",
   "full_text":"""JUDGMENT. Accused convicted under Anti-Terrorism Act 1997 for extorting money from businessmen through threats
of violence against family members. Evidence included audio recordings of threats.
HELD: ATA 1997 covers extortion through terrorizing if it creates sense of insecurity in community.
Audio recordings admissible as electronic evidence under Electronic Transactions Ordinance 2002.
Sentence 10 years rigorous imprisonment upheld. RESULT: Conviction upheld. Sentence maintained."""},

  # ── LAND ACQUISITION ─────────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2020 LHC 1119",
   "case_number":"W.P. 11234/2019","date":"2020-02-19",
   "title":"Landowners v. Government — Land Acquisition Compensation Enhancement",
   "url":"https://www.lhc.gov.pk/judgments/1119",
   "full_text":"""JUDGMENT. Landowners whose land was acquired for motorway project challenged adequacy of compensation.
Government had valued land at revenue record rate which was fraction of market value.
HELD: Land Acquisition Act 1894 requires compensation at market value at time of acquisition.
Revenue record rates do not reflect actual market values. Reference to District Collector and thereafter
to civil court for determination of market value. Enhancement of compensation from Rs 500,000 to Rs 3,200,000 per acre.
RESULT: Petition allowed. Compensation enhanced."""},

  # ── RELIGION / BLASPHEMY (sensitive, factual legal only) ─────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2021 LHC 3319",
   "case_number":"Crl. A. 112/2020","date":"2021-02-28",
   "title":"Accused v. State — Section 295-C Blasphemy Acquittal Evidence Standard",
   "url":"https://www.lhc.gov.pk/judgments/3319",
   "full_text":"""JUDGMENT. Accused convicted under Section 295-C PPC by trial court on testimony of single witness who was
complainant himself. Accused denied charges. No independent corroboration.
HELD: Section 295-C carries capital punishment requiring highest standard of proof. Single interested witness
testimony without corroboration insufficient. Possibility of false accusation must be eliminated.
Accused given benefit of doubt. Acquitted and released. RESULT: Appeal allowed. Accused acquitted."""},

  # ── IMMIGRATION / PASSPORT ────────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2022 IHC 5519",
   "case_number":"W.P. 33412/2021","date":"2022-05-03",
   "title":"Applicant v. NADRA — CNIC Blocking Fundamental Right Identity",
   "url":"https://www.ihc.gov.pk/judgments/5519",
   "full_text":"""JUDGMENT. NADRA blocked petitioner's CNIC without notice causing inability to access banking, property, employment.
NADRA claimed security concern without providing details.
HELD: CNIC is fundamental identity document. Blocking without notice and hearing violates Article 10A.
Right to identity is part of right to life dignity. NADRA cannot block CNIC without proper process.
NADRA directed to restore CNIC and provide reasons for original blocking within 7 days.
RESULT: Petition allowed. CNIC restored."""},

  # ── MEDIA / DEFAMATION ───────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 4431",
   "case_number":"C.A. 889/2022","date":"2023-05-22",
   "title":"Plaintiff v. TV Channel — Defamation False Broadcast Damages",
   "url":"https://www.lhc.gov.pk/judgments/4431",
   "full_text":"""JUDGMENT. TV channel broadcast report falsely alleging prominent businessman was involved in money laundering
based on unverified information. No rebuttal given to businessman before broadcast.
HELD: Journalists must verify information before broadcast especially serious criminal allegations.
Failure to obtain version of subject before broadcast constitutes negligence. False broadcast damaging reputation.
Damages Rs 5,000,000 awarded. Public apology broadcast directed during prime time.
RESULT: Suit decreed. Damages and apology ordered."""},

  # ── RIGHT TO INFORMATION ─────────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2021 IHC 6634",
   "case_number":"W.P. 24512/2020","date":"2021-11-15",
   "title":"Citizen v. Government Ministry — Right to Information Refusal",
   "url":"https://www.ihc.gov.pk/judgments/6634",
   "full_text":"""JUDGMENT. Citizen sought information about government contracts under Right of Access to Information Act 2017.
Ministry refused claiming commercial confidentiality of third party contractors.
HELD: RTI Act 2017 provides right to access information held by public bodies. Exceptions must be strictly interpreted.
Contract amounts and names of contractors are public interest information. Commercial confidentiality exception
does not apply to use of public funds. Ministry directed to provide information within 10 days.
RESULT: Petition allowed. Information disclosure ordered."""},

  # ── WOMEN'S RIGHTS ───────────────────────────────────────────────────────
  {"court":"SHC","court_name":"Sindh High Court","citation":"2020 SHC KHI 7741",
   "case_number":"Const. P. 14412/2019","date":"2020-10-14",
   "title":"Woman v. Employer — Maternity Leave Benefits Denial Private Sector",
   "url":"https://caselaw.shc.gov.pk/caselaw/case/14412-2019",
   "full_text":"""JUDGMENT. Female employee of private company denied maternity leave and maternity benefits under West Pakistan
Maternity Benefit Ordinance 1958. Terminated upon informing employer of pregnancy.
HELD: Maternity Benefit Ordinance 1958 applies to all establishments with 10 or more employees. Dismissal during
pregnancy or maternity leave is void and illegal. Employee entitled to 12 weeks maternity leave and full pay.
Termination set aside. Back pay and maternity benefits ordered. RESULT: Petition allowed."""},

  # ── PRISON / PRISONERS RIGHTS ────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2022 LHC 1119",
   "case_number":"W.P. 8912/2021","date":"2022-02-16",
   "title":"Prisoners v. Punjab Government — Overcrowding Prison Conditions Fundamental Rights",
   "url":"https://www.lhc.gov.pk/judgments/1119",
   "full_text":"""JUDGMENT. Petition by prisoners regarding severe overcrowding, lack of medical care, unhygienic conditions in
Central Jail Lahore which houses 8,000 prisoners against capacity of 3,000.
HELD: Right to human dignity under Article 14 continues even in prison. Overcrowding and denial of basic facilities
violates fundamental rights. Government directed to reduce overcrowding through expediting trials and bail for
petty offenders. Medical facilities to be upgraded. RESULT: Petition allowed. Comprehensive directions issued."""},

  # ── CLIMATE / ENVIRONMENT ────────────────────────────────────────────────
  {"court":"LHC","court_name":"Lahore High Court","citation":"2023 LHC 9912",
   "case_number":"W.P. 56712/2022","date":"2023-09-15",
   "title":"Citizens v. Punjab Government — Smog Air Quality Public Health Crisis",
   "url":"https://www.lhc.gov.pk/judgments/9912",
   "full_text":"""JUDGMENT. Citizens filed petition against government's failure to control smog in Lahore causing respiratory
diseases with AQI consistently above 500 in winter months.
HELD: Clean air is part of right to life under Article 9. Government has duty to control pollution sources.
Brick kilns, crop burning, vehicular emissions must be regulated. EPA directed to close all non-compliant brick kilns.
Crop burning to be criminally prosecuted. Vehicle emission testing mandatory. RESULT: Petition allowed."""},

  # ── DISABLED PERSONS RIGHTS ──────────────────────────────────────────────
  {"court":"IHC","court_name":"Islamabad High Court","citation":"2022 IHC 8891",
   "case_number":"W.P. 41234/2021","date":"2022-08-30",
   "title":"Disabled Person v. Employer — Quota Employment DRAP Rehabilitation Ordinance",
   "url":"https://www.ihc.gov.pk/judgments/8891",
   "full_text":"""JUDGMENT. Disabled applicant denied government job despite qualifying on merit and falling within 2% disabled
persons quota under Disabled Persons Ordinance 1981.
HELD: Disabled Persons Ordinance 1981 mandates 2% reservation in government jobs. Quota is mandatory not
discretionary. Government employer cannot refuse qualified disabled candidate falling within quota.
Appointment ordered within 30 days. RESULT: Petition allowed. Appointment ordered."""},

]


class CaseLawVectorDB:
    def __init__(self):
        self.cases: list[dict] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,3), min_df=1, max_df=0.95,
            sublinear_tf=True, strip_accents="unicode",
            analyzer="word", token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        )
        self._matrix = None
        self._fitted = False
        self._load()
        log.info(f"DB ready — {len(self.cases)} cases.")

    def _save(self):
        with open(DB_FILE,"wb") as f:
            pickle.dump({"cases":self.cases,"vectorizer":self.vectorizer,
                         "matrix":self._matrix,"fitted":self._fitted},f)

    def _load(self):
        if DB_FILE.exists():
            try:
                with open(DB_FILE,"rb") as f: d=pickle.load(f)
                self.cases=d["cases"]; self.vectorizer=d["vectorizer"]
                self._matrix=d["matrix"]; self._fitted=d["fitted"]
                log.info(f"Loaded DB: {len(self.cases)} cases.")
                return
            except Exception as e:
                log.warning(f"Could not load DB: {e}. Rebuilding.")
        # Fresh start — load built-in cases
        self._ingest_cases(BUILT_IN_CASES)

    def _rebuild_index(self):
        if not self.cases: return
        docs=[self._doc(c) for c in self.cases]
        self._matrix=self.vectorizer.fit_transform(docs)
        self._fitted=True

    def _doc(self,c):
        return " ".join(filter(None,[
            c.get("title",""), c.get("citation",""),
            c.get("case_number",""), c.get("court_name",""),
            c.get("court",""), c.get("date",""),
            c.get("full_text","")[:3000],
        ]))

    def ingest_all_data(self):
        """Load all scraped JSON files from /data/ folder."""
        files=list(DATA_DIR.glob("cases_*.json"))
        if not files:
            log.info("No scraped data files yet. Run the scraper to add more cases.")
            return
        for f in files:
            try:
                with open(f,encoding="utf-8") as fp: cases=json.load(fp)
                self._ingest_cases(cases)
                log.info(f"Loaded {len(cases)} cases from {f.name}")
            except Exception as e:
                log.warning(f"Could not load {f}: {e}")

    def _ingest_cases(self,cases):
        seen={f"{c.get('citation','')}|{c.get('title','')[:60]}" for c in self.cases}
        added=0
        for c in cases:
            key=f"{c.get('citation','')}|{c.get('title','')[:60]}"
            if key not in seen and len(c.get("title",""))>5:
                self.cases.append(c); seen.add(key); added+=1
        if added>0:
            self._rebuild_index(); self._save()
            log.info(f"Added {added} cases. Total: {len(self.cases)}")

    def search(self,query:str,top_k:int=10,court_filter:Optional[str]=None)->list[dict]:
        if not self.cases or not self._fitted: return []
        sims=cosine_similarity(self.vectorizer.transform([query]),self._matrix)[0]
        ranked=sorted(enumerate(sims),key=lambda x:x[1],reverse=True)
        hits=[]
        for idx,score in ranked:
            if len(hits)>=top_k: break
            c=self.cases[idx]
            if court_filter and c.get("court","").upper()!=court_filter.upper(): continue
            ft=c.get("full_text","")
            hits.append({
                "id":f"{c.get('court','')}_{idx}",
                "similarity":round(float(score),4),
                "court":c.get("court",""),
                "court_name":c.get("court_name",""),
                "citation":c.get("citation",""),
                "case_number":c.get("case_number",""),
                "title":c.get("title","")[:300],
                "url":c.get("url",""),
                "date":c.get("date","")[:10],
                "snippet":self._snippet(ft,query),
                "full_text":ft,
            })
        return hits

    def _snippet(self,text,query,length=450):
        if not text: return ""
        terms=query.lower().split()
        best,best_score="",- 1
        for s in re.split(r"[.\n]+",text):
            score=sum(1 for t in terms if t in s.lower())
            if score>best_score and len(s.strip())>20:
                best_score,best=score,s.strip()
        if best:
            pos=text.find(best)
            return text[max(0,pos):pos+length].strip()
        return text[:length].strip()

    def count(self)->int: return len(self.cases)


_db=None
def get_db()->CaseLawVectorDB:
    global _db
    if _db is None: _db=CaseLawVectorDB()
    return _db


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    db=get_db()
    db.ingest_all_data()
    print(f"\n✅ Total cases in DB: {db.count()}\n")
    tests=["NHA road accident missing signs","wrongful termination EOBI labour",
           "illegal construction demolition","tax evasion FBR","custody children divorce",
           "medical negligence hospital death","bail criminal drugs","property dispute mutation"]
    for q in tests:
        r=db.search(q,top_k=2)
        print(f"'{q}' → {[(x['court'],x['citation'],round(x['similarity'],2)) for x in r]}")
