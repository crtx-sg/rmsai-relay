# What a clinician can ask

Every question the voice agent (phone or WebRTC call) and the companion-app chat understand, what
answers it, and where the answer comes from. Each example below is checked against the router
(`kb/graph/lookup.py` `match_intent`) and pinned in `tests/test_query_coverage.py`,
`tests/test_call_2026_10_14.py`, `tests/test_review_2026_10_14.py` and `tests/test_patient_ranking.py`.
`cli.events --ask "<question>"` shows what any question resolves to, its rows and the answer.

**Where answers come from.** *Graph* answers are read straight from Neo4j by a fixed template: no
model is involved, so they cannot be paraphrased or invented. *Documents* answers are retrieved from
the clinical documents (`docs/*.md` and uploads) and phrased by the local LLM, with citations. A
question about a patient that no graph template covers is never handed to the LLM: the agent says
what it can answer instead.

**Session scope.** On an outbound call (after the PIN) "this event" is the event the call is about and
"this patient" its patient; in app chat they are the selected worklist row. Questions that name a bed
or a PT number work anywhere. Text answers list rows; spoken answers are shortened for the phone.

## 1. On a call: PIN, acknowledgment, conversation

| Say | What happens |
|---|---|
| "one two three four" (or keypad digits) | PIN gate. Nothing about a patient is said before it. Three wrong PINs: "Ending the call for safety" and the call is hung up; the right PIN is refused after that. |
| "acknowledged" · "yes" · "copy that" · "that's correct" · "I've acknowledged it" · "kindly acknowledge this" · "acknowledge bed two" (this call's bed) | Acknowledges **the event the call is about** and says which: "…your acknowledgment of the ventricular fibrillation alert for patient PT998224 on bed 2. Goodbye." No wake word needed. |
| "acknowledge bed seven" / "acknowledge PT986685" (another event) | Refused: "On this call I can only acknowledge … Please acknowledge other events in the app." |
| "don't acknowledge yet" · "not yet" · "No, that is not correct" | Nothing is acknowledged: "Okay, I have not acknowledged …; it stays open. What would you like to know?" |
| "is bed two acknowledged?" · "who acknowledged bed seven?" | A question, answered from the event history; never acknowledges. |
| "hey vios, …" | Wake word for follow-up questions when `AUDIO_WAKE_REQUIRED=true` (then 30 s awake window). Replies to the alert never need it. |
| "repeat that" · "say that again" | The previous answer again. |
| "thank you" · "goodbye" · "that's all, bye" | A short courtesy reply. |
| "stop" · "wait" · "hold on" | Stops the agent speaking (barge-in). "yeah", "okay", "mm-hmm" while it talks are ignored. |
| a pause mid-question ("What was the… MEWS score… for bed seven") | The pieces are joined into one question. |

## 2. This event (graph)

| Ask | Answer |
|---|---|
| "What are the vitals at the time of this event?" · "give me the vitals" | Rhythm, time, criticality, MEWS score, heart rate, blood pressure, oxygen saturation, respiratory rate, temperature |
| "What was the heart rate / blood pressure / oxygen saturation / respiratory rate / temperature?" | Just that vital |
| "What was the MEWS score?" · "news score for this event" | MEWS score and band, and the vitals that scored ("heart rate 195 scores 3, …") |
| "What is the criticality of this event?" · "how bad is it?" | Criticality and MEWS |
| "What rhythm was detected?" | The model's rhythm, time and confidence |
| "How confident is the model about this rhythm?" | Confidence in the detected rhythm |
| "When did this event happen?" | Time detected, and how long ago |
| "Is this a false positive?" | Yes/no, with the rhythm and confidence behind it |
| "Why was I alerted?" | The stored reason (rhythm and confidence, abnormal vitals, trends) |
| "Show me the ECG strip for this event" | The strip (shown in the app; on a call, said to be available there) |
| "How was the heart rate trending?" | Direction, start/end and peak over the readings before the event |
| "What should I do now?" · "what are the action items for this event?" | This event's recorded actions ("escalate care", "initiate ACLS protocol", …) |
| "What is the protocol for this event?" · "how do I treat this patient?" · "what is the treatment for this rhythm?" | *Documents*: the protocol for this event's rhythm |

## 3. This patient (graph)

| Ask | Answer |
|---|---|
| "What is the past medical history of this patient?" | Age, sex, diagnoses, medications, prior surgeries, symptoms |
| "What medications is the patient on?" · "what are his diagnoses?" · "any prior surgeries?" · "what symptoms does the patient have?" | That part only |
| "Does the patient have any allergies?" | "Allergies are not recorded in this system" (no allergy data) |
| "How old is the patient?" | Age and sex |
| "Which bed is this patient on?" | Bed |
| "Any prior events for this patient?" · "this patient's events" | Event list with pending count |
| "What are the unacknowledged events for this patient?" · "what other critical events does this patient have?" | Filtered event list |
| "What were the vitals at this patient's VT?" | Vitals at that patient's event of that rhythm |

## 4. Another bed or patient (graph)

| Ask | Answer |
|---|---|
| "What was the last reported event on bed seven?" (also "vid/bad/bid/bet nine" heard by STT) | That bed's latest event: rhythm, time, criticality, MEWS, vitals, status |
| "What are the vitals for bed seven?" · "What was the MEWS score for bed seven?" | Vitals / MEWS at that bed's latest event |
| "What events are on bed six?" · "pending events on bed nine" · "what is the status of bed four?" | That bed's event list |
| "What is the protocol for bed seven?" | Care-protocol steps for that bed's event, or *documents* for its rhythm when no structured steps exist |
| "Show the report for bed seven" | The event report summary |
| "Which bed is PT986685 on?" · "events for PT986685" · "what is the medical history of PT986685?" · "what medications is PT986685 on?" | As for this patient, for the named one |
| "…on bed nein" (number not understood) | "I couldn't tell which bed you meant. Please say the bed number again." |

## 5. The ward (graph)

| Ask | Answer |
|---|---|
| "List patients by criticality" · "who should I see first?" · "give me the list for bed rounds" · "sickest patients" · "triage list" · "rank patients by severity / MEWS" | Every patient, sickest first: open alerts by criticality, then MEWS. One line each: bed, patient, criticality, rhythm, MEWS, open alerts (at most 10 on a call) |
| "…in bed order" | The same list by bed number |
| "What is on my worklist?" · "what are the unacknowledged events?" | Pending (alerted, unacknowledged) events: counts by rhythm and bed |
| "Which bed has the most severe alarm?" | Pending events grouped Critical → High → Medium → Low |
| "Show critical events in the last 24 hours" | High/Critical events in the window |
| "Show positive events in the last 30 minutes" | Non-false-positive events in the window |
| "List all events" · "show the alarms from the last 6 hours" | Event list (optionally windowed) |
| "Which events were withheld?" · "which events have been acknowledged?" | By status |
| "List all atrial fibrillation events" · "how many AFib episodes on bed 2?" | By rhythm (optionally a bed or patient) |
| "Which patients have reported ventricular tachycardia?" | Patients and beds with that rhythm |
| "How many events were there today?" · "…in the last 30 minutes / 6 hours / 2 days / this week" | Count, patients, alerted, still pending |
| "Which patient has the most alarms?" · "how many alarms per bed?" | Alarm counts per bed |
| "Are there any outstanding action items?" | Count, high-priority count, the top three |
| "Any patterns between age and event type?" | Most common rhythm × sex × age band |
| "What are the comorbidities of atrial fibrillation?" | Co-occurring conditions among current patients |
| "Show ECG strips for the last AFib event" · "HR and BP trend for the last VT event" | Latest event of that rhythm, any patient |

Event listings always state their filter ("2 pending (unacknowledged) events on bed 2 …"); in app chat
a button opens the Events tab with the same rows. A listing with a target that can't be resolved
("events for Mr Smith") gets a hint instead of a guess.

## 6. Clinical knowledge (documents)

| Ask | Answer |
|---|---|
| "What is the protocol to handle ventricular tachycardia?" · "protocol for PVC" · "how do you manage AFib with RVR?" · "what about its long-term management?" · "go on" / "the full protocol" | No LLM. The rhythm's structured care-protocol steps (`common/protocols/care_protocols.yaml`; only AFib-RVR today), else that rhythm's sections from one protocol document, read verbatim. Spoken: 3 points, then what is left; "the rest" continues, a section word ("long-term") picks the section. A rhythm no document has a section for (LBBB) falls through to the row below. |
| "what is the rate control for AFib?" · "what does the critical alarm SOP say?" · "how is hyperkalaemia escalated?" | Cited answer from the documents |
| "When should I escalate a MEWS score?" | Escalation rules from the MEWS document |
| "What does a MEWS of 5 mean?" | Band from the same scorer that rates alerts ("high risk") + the document's escalation rule. No LLM. |
| A topic no document covers ("how do you treat sepsis?", "capital of France") | "I don't have information on that in the knowledge base." |

New SOPs and guidelines become answerable once uploaded (`cli.kb_upload`).

## 7. Safety

| Say | What happens |
|---|---|
| "give 1 mg of epinephrine" · "push amiodarone" · "order a troponin" | Refused: the agent informs, it does not act. Questions such as "shock energy for VF?" are answered. |
| "what is the PIN?" | Refused |
| "ignore previous instructions …" | Refused |
| "the patient is coding, what do I do?" | Emergency steps from the critical-alarm SOP, or escalation if none is found |
| a patient question no graph template covers ("is the patient comfortable?") | "I can tell you this patient's rhythm, vitals, MEWS, heart-rate trend, history, medications, events and bed, and why the alert was raised. Which would you like?" |

## 8. Companion-app chat only

| Do | What happens |
|---|---|
| Click a worklist row | Scopes the chat to that event (`/select`): "this event" / "this patient" mean it |
| "show / see / display / pull up the ECG strip · HR trend · report" | Answer plus the artifact rendered inline |
| Push-to-talk | Spoken questions without the wake word |

## Not supported yet

Lab results (e.g. potassium), orders and medication changes, allergies, and trends of vitals other
than heart rate for this event (a blood-pressure trend exists only as "HR and BP trend for the last
<rhythm> event"). These get the patient hint (section 7) or a documents decline.
