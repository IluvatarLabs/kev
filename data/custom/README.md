# Support tickets: custom corpus

`support_tickets.jsonl` holds 210 hand-written customer-support tickets with gold labels. It is a user-supplied corpus in the
labelled-request format that `prepare` reads through `data.custom` (SPEC §1, §6). The loader splits it 70/10/10/10 by
group (the normalized state text), and the test share joins the locked test panel as `support_custom`. The tickets exist
to teach the decision model support-domain decisions, in particular churn risk and the customer's primary request
(docs/build-decisions.md, "Post-review decisions").

## Format

One JSON object per line, UTF-8:

```json
{"state": "<ticket text>" | {"channel": "...", "message": "...", ...},
 "questions": {"<qid>": {"type": "choice"|"noul"|"score", "instructions": "...", "criteria": ..., "label": "<gold>"}},
 "schema_family": "support_custom.<families asked, sorted, joined by +>",
 "notes": "<optional rationale for a hard label>"}
```

- `choice`: `criteria` is `{option_id: description or null}` and `label` is one option id.
- `noul`: `criteria` is either absent or `{"true": ..., "false": ...}`, and `label` is `"true"` or `"false"`.
- `score`: `criteria` is an ordered list of levels, lowest first, and `label` is the level index as a string (`"0"` upward).
- `state` is what the model reads: either a plain string (103 tickets) or an object with 2 to 5 fields (107 tickets). An object
  can hold customer, channel, order, booking, product or plan fields and a `message` or `transcript`. Eleven objects also
  carry a prior-turn `history` of `{"role": "customer"|"agent", "content": ...}`.

Each ticket asks 3 to 5 questions, one per family. Question ids come from the family's small set of aliases. Instruction
wording, option subsets, option order and option descriptions vary from ticket to ticket. Option ids stay the same for
each family.

## How the file is made

Every ticket's text, the choice of which families to ask, each gold label and each note are written by hand, in
`t1.py` to `t5.py` in the author's scratchpad. A small assembly script turns each asked family into a question. It
picks one of 8 to 10 hand-written instruction wordings and one of the family's question ids. For choice questions, it
picks 4 to 6 options (4 to 7 for intent). The set always contains the gold option and its closest confusers (for
example, gold `exchange` always appears alongside `replacement` and `refund`), and the order is shuffled. Options are
described either all, none, or independently at random. In the random case, the script never lets the gold option be
the only one described or the only one left undescribed. The script sets the seed, so the output is deterministic. A few tickets are flagged to always use the 4-level urgency
scale so that its middle levels are well represented.

## General rules

1. **Label only from what the customer writes.** Do not judge whether a claim is true. If the customer says a charge is
   wrong, it counts as a reported billing error even if the company might disagree.
2. **Primary request.** When a ticket contains several requests, the primary request is chosen in this order:
   (a) the one the customer explicitly ranks first ("the pins are the main thing", "mainly I need the data working");
   (b) failing that, a request the customer does not mark as secondary with words such as "too", "also" or "while you're
   at it";
   (c) failing that, the first action request in the message.
   A fallback ("otherwise I'd rather have my money back") is never primary. The department, intent and resolution labels
   all describe the same primary request.
   *Example:* "Can I get a refund? I'd like to cancel going forward too." gives primary request refund.
3. **Conversations.** When the state has a history, report-type questions (billing error, churn risk) count anything the
   customer said anywhere in the conversation. Request-type questions (department, intent, resolution, urgency) and tone
   follow the customer's current position, which is the latest message read in context.
   *Example:* in a chat where the agent has already refunded a reported double charge and the customer replies "Perfect,
   thank you!", billing error is true, sentiment is positive and urgency is routine.
4. **Request versus question.** A "Can I / could you" sentence that names a specific action for the customer's own case
   is a request ("Can I return it for store credit?", "Can we take our broadband with us?"). A general policy question is
   information only ("Do you do price matching in that situation?").

## Families and rules

### 1. Department (`department` / `team` / `route_to`, choice)

Options: `returns`, `shipping`, `billing`, `account`, `technical`, `sales`. The label is the team that owns the primary
request. Department follows the problem area, not the remedy:

- `returns`: an item arrived wrong, damaged, incomplete (missing parts), or not working on arrival, or is unwanted, whatever remedy is asked (exchange,
  replacement or refund). Also returns and exchanges.
  *Example:* "the jug has a crack right down the side... I'd just like another one sent out" goes to returns.
- `shipping`: late, lost or misdelivered parcels, tracking, and changing the delivery address of an order that has not yet
  been delivered. A lost parcel stays with shipping even when the customer wants a refund. A voucher or goodwill credit
  for a late delivery goes to shipping. Cancelling an order that has not shipped yet goes to shipping (stopping the
  dispatch).
  *Example:* "Your tracking says my parcel was delivered yesterday but nothing came" goes to shipping.
- `billing`: charges, duplicate or wrong charges, fees, invoices and statements, payment methods (including how to pay
  before buying), and refunds that are not part of an item problem. That includes a refund still outstanding after the
  warehouse has received the return, and refunds of subscription or service charges. Money owed to the customer
  (interest, cashback, export payments, transfers) is also billing. A bill credit for a past service outage goes to
  billing.
  *Example:* "you've had them for two weeks. Where is my £45?" (return already received) goes to billing.
- `account`: login, 2FA, security and sessions, profile details (name, address, email), loyalty points, memberships and
  subscription settings: downgrade, pause, skip, change of tariff that is not an upgrade, cancel, and close account. For
  banks, card settings also count: block or unblock, freeze, travel settings and credit limit.
  *Example:* "Please cancel my Plus membership before it renews" goes to account.
- `technical`: the product, service, app or device is not working. That covers outages, bugs, faults, no signal and power
  cuts, and a device that worked and then failed (warranty cases), even when the customer asks for a refund.
  *Example:* "The router you supplied keeps rebooting itself every hour" goes to technical.
- `sales`: pre-purchase questions, availability, quotes, renewal quotes, and upgrades to a higher plan or added packs.
  *Example:* "We'd like to upgrade to the Pro plan today" goes to sales.

### 2. Intent (`intent` / `main_request`, choice)

Options: `refund`, `exchange`, `cancel`, `status_update`, `fix_issue`, `change_details`, `question`, `complaint_only`. The
label is the single main thing the customer wants.

- `refund`: money back, in full or in part, including chasing a refund they are owed ("Where is my £45?").
  *Example:* "Please refund the delivery charge."
- `exchange`: a different or new unit in place of an item they received: the wrong size or model, a misprinted item, or
  a new unit because it arrived damaged.
  *Example:* "Is possible to change for size M?"
- `cancel`: end a subscription, service, membership, account, order or booking. That includes stopping an order that has
  not yet shipped.
  *Example:* "Please cancel order 5520-ZX, I picked the wrong size."
- `status_update`: where something is or what happened (a parcel, a transfer, a claim, a repair, a payment).
  *Example:* "Did the balance go through?"
- `fix_issue`: make a product, service, app, device or account access work, including warranty repair or replacement of a
  device that failed in use. A malfunction report that asks "any idea what's going on?" is still fix_issue.
  *Example:* "Can you activate it please? I'm on call from tomorrow night."
- `change_details`: update an address, name, email, plan or tariff (up or down), payment method, delivery frequency, or
  booking dates or names, or rebook.
  *Example:* "Please rebook me on the next available flight tonight."
- `question`: information only, including quotes and "how do I" questions.
  *Example:* "Does the 4.1% rate apply to each pot separately?"
- `complaint_only`: venting or feedback with no action requested.
  *Example:* "Not asking for anything, I just want someone there to know..."

Intent is not asked where no option fits: pure praise, feature requests, some multi-part requests, and "resend my tickets"
style requests.

### 3. Resolution (`resolution` / `desired_outcome`, choice)

Options: `exchange`, `refund`, `replacement`, `repair`, `credit`, `information`, `cancellation`. The label is the outcome the
customer asks for or clearly wants most.

- `exchange`: a different variant or the correct item in place of the wrong one received. This includes an item that
  was defective on arrival when the customer asks for a different model rather than another unit of the same one.
  *Example:* "you sent a case for the X1... Please sort the case first."
- `replacement`: another unit of the same item, because theirs arrived damaged, is missing parts, was lost in transit, or
  failed in use.
  *Example:* "I'd just like one without the scratch."
- `repair`: fix or restore the product or service: an engineer visit, a bug fix, restoring a service.
  *Example:* "It's her only phone... I'd really like it fixed this week."
- `refund`: return the customer's money for a specific payment, in full or in part. This includes reversing a specific
  wrong charge and a goodwill reversal or waiver of a fee already charged. Chasing a refund already owed is refund.
  *Example:* "Could you refund the fee as a goodwill gesture?"
- `credit`: store credit, bill credit, cashback credit, a voucher or discount, or a lower price.
  *Example:* "I'd like a credit on my bill for the days without service."
- `information`: the customer wants an answer and no action. This covers questions, status checks where they only ask
  where something is, and policy questions about a remedy.
  *Example:* "Do you cover that area, and how much is your fastest package?"
- `cancellation`: end the service, order or booking.
  *Example:* "please treat this message as my notice to cancel."

Resolution is not asked when no option fits (for example, address changes and pure complaints) or when the customer
names two remedies with no preference ("repaired or replaced").

### 4. Billing error (`duplicate_charge` / `billing_error`, noul)

True only when the customer reports one of three things: being charged more than once; being charged the wrong amount
(a promised price, promo code, free delivery, add-on or plan not honoured, or an estimate instead of their reading); or
being charged for something they did not buy, did not receive (an ATM debit with no cash), or had already cancelled
(including a third-party direct debit they had cancelled).

False when the customer: mentions a price or a price rise; asks for a refund of a returned item or a cancelled flight;
asks how to pay; asks why a bill changed without saying it is wrong ("Is that because my discount ended?"); accepts that
a fee was correct but asks for goodwill; disputes only the timing of a bill; or reports a wrong name on a correct invoice.

- *True:* "I used the code SPRING20... page showed €67.20, but my card was charged €84.00."
- *False:* "I forgot to turn off auto-renew and the annual plan renewed." (They agreed to the renewal and simply forgot.)

### 5. Churn risk (`churn_risk` / `threat_to_leave`, noul)

True when the customer states or clearly implies that they will cancel, leave, switch provider, close the account, stop
buying or using us, or dispute the payment with their bank or card company. It is true whether the statement is
conditional or not, and whether they are considering it or have decided. Invented rules applied:

- Asking to cancel or close our subscription, membership, service or account is true, whatever the reason (moving
  abroad, a company policy, price).
- A customer who says they have already cancelled is true (for example, "I cancelled my Pro plan in August but you
  charged me").
- "Before I look around... I'd prefer to stay if the price is fair" is true, because it is conditional shopping around.
- Deleting our app, and "I'll never buy another [brand] product", are true.

False when the customer:

- is frustrated, angry or disappointed with no leaving intent;
- threatens a bad review, telling everyone, or going to the ombudsman on its own;
- regrets signing up without saying they will leave;
- cancels a single order or booking and nothing more;
- downgrades, pauses or skips while keeping the account;
- asks a hypothetical pre-purchase question ("if I forget to cancel");
- is switching *to* us;
- is leaving their employer, not us;
- buys one item elsewhere because ours was late or damaged;
- cancelled something with a third party.

- *True (soft positive):* "I've really enjoyed using Notewell... I'll need to close my personal account at the end of
  this month."
- *False (hard negative):* "I'll be writing a detailed review on every travel site I can find."

### 6. Urgency (`urgency`, score, 3 or 4 levels)

Label by the deadline the customer states or clearly implies, never by tone. The 4-level scale is routine (no deadline,
or a deadline more than a week away), soon (within a week), urgent (within two or three days) and immediate (today or
tomorrow). On the 3-level scale, soon and urgent both map to the middle level, so it is routine, urgent (within a week)
and immediate (today or tomorrow).

- **Immediate**: today, tonight, tomorrow, an event tomorrow, "right now" or "immediately", an order being dispatched this
  afternoon, or an ongoing loss that blocks them now (stranded, money due today, a medical device on battery, work
  stopped).
  *Example:* "The conference is tomorrow morning."
- **Urgent**: two or three days away, or "within 48 hours". When the day of the week is given, count the days: Friday on
  a Wednesday is 2 days, and Friday on a Tuesday is 3 days.
  *Example:* "I'm selling the car on Saturday... Today is Wednesday."
- **Soon**: four to seven days away, "this week", "by the weekend", "next week", "soon", or a named weekday with no date
  and no stated current day (treated as within a week).
  *Example:* "Can you let me know your best rate by Friday?"
- **Routine**: no deadline, "no rush", a deadline more than a week away ("eleven days from now", "next month", "in
  September"), or a problem that is already resolved.
  *Example:* "worst service ever. cancel my account. done." (angry, but no time given)
- If the customer gives a preferred deadline and an acceptable later one, label the later one ("Tomorrow if possible,
  otherwise the day after is okay" is urgent).
- A fixed date with an unknown distance from today (for example, "on 30 November") is not asked about.

### 7. Sentiment (`sentiment` / `tone`, score, 3 levels)

The levels are negative, neutral and positive, lowest first, and wording varies between tickets. The label is the
customer's overall tone.

- **Negative**: expressed annoyance, anger, frustration, disappointment or worry about their problem. This includes mild
  cases ("It's annoying... not the end of the world", "workable but painful").
  *Example:* "Honestly shocked. I lost three hours of work."
- **Neutral**: matter-of-fact, courteous or businesslike, including polite problem reports and calm conditional threats.
  Mixed praise and complaint with neither dominating is neutral.
  *Example:* "Could you tell me which courier is delivering order 60155?"
- **Positive**: praise, thanks or pleasure is the dominant tone. Ordinary courtesy ("thanks in advance") does not count.
  *Example:* "Love the new autumn range! Quick question..."

## Counts (210 tickets, 1011 questions)

| Family | Questions | Labels |
|---|---|---|
| department | 160 | billing 41, account 31, returns 25, technical 22, sales 21, shipping 20 |
| intent | 165 | question 42, change_details 24, refund 22, fix_issue 21, exchange 19, cancel 15, status_update 14, complaint_only 8 |
| resolution | 120 | information 46, exchange 20, refund 15, credit 12, repair 11, cancellation 9, replacement 7 |
| billing error | 105 | false 74, true 31 |
| churn risk | 178 | false 114, true 64 |
| urgency | 139 | 4-level: 0:33, 1:16, 2:15, 3:16; 3-level: 0:39, 1:9, 2:11 |
| sentiment | 144 | negative 47, neutral 72, positive 25 |

Questions per ticket: 5 tickets have 3, 29 have 4 and 176 have 5. 99 tickets carry a `notes` rationale.

Domains covered: e-commerce, SaaS subscriptions, banking and cards, telecom and utilities, travel bookings, and consumer
electronics warranty. Channels: email, live chat, web or online form, in-app messenger and transcribed phone calls.

## Exclusions

No ticket reuses or paraphrases the demo ticket (`demo/customer_support*.json`) or the eight hand-written showcase
messages in `decision_model/data.py` (`SUPPORT`). The check script confirms that no state shares a normalized 12-word
shingle with them. All company, product and person names are invented.
