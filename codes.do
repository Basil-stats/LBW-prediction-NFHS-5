*==============================================================================
* Predicting low birth weight from antenatal information using machine learning:
* Evidence from the National Family Health Survey-5, India
*==============================================================================

* ----paths----
global rawdata "C:/path/to/IAKR7EFL.DTA"
global outdir  "C:/path/to/output"

capture log close
log using "$outdir/LBW_revision_log.smcl", replace

* start from the raw file every time
use "$rawdata", clear

* NEW: keep only the most recent birth
keep if midx == 1

* survey weight and design
gen wt = v005/1000000
svyset v021 [pw=wt], strata(v022) singleunit(centered)


************* Dependent variables *****************

* birth weight measured
drop if (m19a==0 | m19a==8 | m19a==.)
gen bw_card = (m19a==1)        
label variable bw_card "Birth weight from health card"


drop if m19 >= 9996 | m19 == .          
drop if m19 > 4500                      
gen bwt = .
replace bwt = 1 if m19 <  2500
replace bwt = 0 if m19 >= 2500 & m19 <= 4500
label define bwt 0 "Normal weight" 1 "Low birth weight"
label values bwt bwt
label variable bwt "birth weight"
ta bwt


***************explanatory variables*****************

** age of mother at birth **
*   child's date of birth minus mother's date of birth
gen aab = int((b3 - v011)/12)
recode aab (min/19=1 "<=19") (20/24=2 "20-24") (25/29=3 "25-29") (30/max=4 ">=30"), gen(aom)
label variable aom "Age of mother at birth"
ta aom


** sex of child **

gen sex = (b4==2) if b4 < .
label define sex 0 "Male" 1 "Female"
label values sex sex
ta sex


** marital status **

recode v501 (1 2=1 "Married/living with partner") (3=2 "widowed") (0 4 5=3 "others"), gen(marsta)
ta marsta


** occupation **

recode v717 (0 98=0 "not working") (1/97=1 "working"), gen(work)
label variable work "work status"
ta work


** religion **

recode v130 (1=1 "Hindu") (2=2 "Muslim") (3=3 "christian") (nonmissing=4 "Others"), gen(religion)
label variable religion "Religion of mother"
ta religion, m


** residence **

recode v025 (2=1 "Rural") (1=2 "Urban"), gen(residence)
label variable residence "type of residence"
ta residence


** education **

clonevar education = v106   // CHANGED: clonevar instead of rename, keeps v106 and its labels
ta education


** wealth **

clonevar wealth = v190
ta wealth


** height **


replace v438=. if v438 >= 9990
gen height=v438/10
summ height


** weight **

replace v437=. if v437 >= 9990
gen weight=v437/10
summ weight


gen preg_now = (v213==1)
replace weight = . if preg_now==1
ta preg_now


** birth order **

recode bord (5/max=5 "5 or more"), gen(bo)
label variable bo "birth order"
ta bo


** birth interval **

recode b11 (0/23=1 "less than 2 yrs") (24/47=2 "2-4 years") (48/max=3 "more than 4 years") ///
           (missing=0 "First birth"), gen(bi)
label variable bi "Preceding birth interval"
ta bi


** tobacco use **

local bidi s708
gen tobacco = .
replace tobacco = 0 if v463a < . | s711 < .
replace tobacco = 1 if v463a==1 | s711==1
capture confirm variable `bidi'
if _rc==0 replace tobacco = 1 if inlist(`bidi', 1, 2)   // every day / some days
else di as error "Bidi variable not found: set local bidi after -lookfor bidi-"
label define tobacco 0 "no" 1 "yes"
label values tobacco tobacco
label variable tobacco "Current tobacco use (at survey)"
gen smoking = tobacco        // keeps the column name the Python notebook expects
ta tobacco, m


** morbidity **


gen chronic = 0
replace chronic = 1 if (s728a==1 | s728b==1 | s728c==1 | s728d==1 | s728e==1 | s728f==1 | s728g==1)
replace chronic = . if s728a==. & s728b==. & s728c==. & s728d==. & s728e==. & s728f==. & s728g==.
label define yesno 0 "no" 1 "yes"
label values chronic yesno
label variable chronic "Chronic disease at survey (Q728)"
ta chronic, m


local q433 m47
local q434 s434
local q435 s435
capture confirm variable `q433' `q434' `q435'
if _rc==0 {
    gen morbidity = 0 if `q433'<. | `q434'<. | `q435'<.
    replace morbidity = 1 if `q433'==1 | `q434'==1 | `q435'==1
}
else {
    di as error "Q433-435 variables not found: set the locals above. Using chronic as a placeholder."
    gen morbidity = chronic
}
label values morbidity yesno
label variable morbidity "Pregnancy complications (Q433-435)"
ta morbidity, m


** ANC visits **

recode m14 (0=0 "no anc visits") (1/2=1 "1-2 anc visits") (3/97=2 "more than 2 anc visits") ///
           (98=.), gen(anc)
label variable anc "no. of anc visits"
ta anc, m


gen anc_first_trim = (m13 <= 3) if m13 < 98
replace anc_first_trim = 0 if m14 == 0
label values anc_first_trim yesno
label variable anc_first_trim "First ANC in first trimester"
ta anc_first_trim, m


** type of delivery ***

recode m17 (0=0 "Normal") (1=1 "caesarian") (8 9=.), gen(tod)
label variable tod "type of delivery"
ta tod


** place of delivery **

recode m15 (21/29=1 "Public") (31/39 41/46=2 "Private/NGO") (11/13 96=3 "Home/Others") ///
           (nonmissing=.), gen(pod)
label variable pod "place of delivery"
ta pod, m


** anemia **

recode v457 (4=0 "non-anemic") (3=1 "mild") (2=2 "moderate") (1=3 "severe") (nonmissing=.), gen(anemia)
label variable anemia "anemia status (at survey)"
ta anemia


** pregnancy intend **

recode m10 (2 3 =0 "Not intended") (1=1 "Intented") (nonmissing=.), gen(pregint)
label variable pregint "wanted pregnancy when became pregnant"
ta pregint


**************** Sample check *********************

count
foreach v in bwt aom sex marsta work religion residence education wealth height weight bo bi ///
             tobacco chronic morbidity anc tod pod anemia pregint {
    quietly count if missing(`v')
    di "`v'" _col(15) "missing: " r(N)
}


** region (6 zones) **

decode v024, gen(state)
replace state = lower(state)
gen region = .
* North
replace region = 1 if strpos(state,"jammu") | strpos(state,"ladakh") | strpos(state,"himachal") | ///
    strpos(state,"punjab") | strpos(state,"chandigarh") | strpos(state,"uttarakhand") | ///
    strpos(state,"haryana") | strpos(state,"delhi") | strpos(state,"rajasthan")
* Central
replace region = 2 if strpos(state,"uttar pradesh") | strpos(state,"madhya") | strpos(state,"chhattisgarh")
* East
replace region = 3 if strpos(state,"bihar") | strpos(state,"jharkhand") | strpos(state,"odisha") | ///
    strpos(state,"orissa") | strpos(state,"west bengal")
* Northeast
replace region = 4 if strpos(state,"arunachal") | strpos(state,"assam") | strpos(state,"manipur") | ///
    strpos(state,"meghalaya") | strpos(state,"mizoram") | strpos(state,"nagaland") | ///
    strpos(state,"sikkim") | strpos(state,"tripura")
* West
replace region = 5 if strpos(state,"goa") | strpos(state,"gujarat") | strpos(state,"maharashtra") | ///
    strpos(state,"dadra") | strpos(state,"daman")
* South
replace region = 6 if strpos(state,"andhra") | strpos(state,"karnataka") | strpos(state,"kerala") | ///
    strpos(state,"tamil") | strpos(state,"telangana") | strpos(state,"puducherry") | ///
    strpos(state,"pondicherry") | strpos(state,"lakshadweep") | strpos(state,"andaman")
label define region 1 "North" 2 "Central" 3 "East" 4 "Northeast" 5 "West" 6 "South"
label values region region
ta state if region == .        // no rows
ta region


**************** Drop missing values *********************

* Model A (antenatal) variables

local modelA aom education wealth religion residence marsta bo bi pregint anc morbidity tobacco height
* Model B adds post-delivery / survey-time variables
local modelB `modelA' weight anemia sex tod pod


replace height = . if height < 130 | height > 200
replace weight = . if weight < 25  | weight > 150
replace anemia = . if anemia >= .     

egen nmiss = rowmiss(bwt `modelB')
drop if nmiss > 0
drop nmiss
count


**************** LBW prevalence **************

ta bwt                                  
svy: proportion bwt                     


****** Table 1 *********

foreach v in aom education wealth religion residence region marsta bo bi pregint anc ///
             morbidity tobacco sex tod pod anemia {
    di _n "======== `v' ========"
    ta `v' bwt                                     // unweighted counts (sample size columns)
    svy: tab `v' bwt, row percent format(%5.2f)    // weighted row %
}


**************** Combined birth order + interval (for regression) **************

ta bo bi
tab b0 bo if bo>=2 & bi==0
gen bobi = .
replace bobi = 1 if bo == 1
replace bobi = 1 if bo >= 2 & bi == 0        // multiple births from a first pregnancy (n = 531)
replace bobi = 2 if inlist(bo, 2, 3) & bi == 1
replace bobi = 3 if inlist(bo, 2, 3) & inlist(bi, 2, 3)
replace bobi = 4 if bo >= 4 & bo < . & bi == 1
replace bobi = 5 if bo >= 4 & bo < . & inlist(bi, 2, 3)
label define bobi 1 "First birth" 2 "Order 2-3, interval <24m" 3 "Order 2-3, interval 24m+" ///
                  4 "Order 4+, interval <24m"  5 "Order 4+, interval 24m+", replace
label values bobi bobi
label variable bobi "Birth order and preceding interval"
ta bobi, m                                    // First birth = 47,556; no missing
svy: tab bobi bwt, row percent format(%5.2f)


**************** Logistic regression *********************

* Model A — antenatal (Table 2)
svy: logistic bwt i.aom i.education i.wealth i.religion i.residence i.region i.marsta ///
     i.bobi i.pregint i.anc i.morbidity i.tobacco c.height

* Model B — full 
svy: logistic bwt i.aom i.education i.wealth i.religion i.residence i.region i.marsta ///
     i.bobi i.pregint i.anc i.morbidity i.tobacco c.height c.weight i.anemia ///
     i.sex i.tod i.pod


**************** Export for the Python *********************

export delimited bwt sex anc marsta morbidity bi tod pod anemia height weight bo pregint ///
       education wealth aom work smoking religion residence ///
       wt v021 v022 region bw_card chronic anc_first_trim preg_now bobi ///
       using "$outdir/data.csv", replace nolabel


count

log close
