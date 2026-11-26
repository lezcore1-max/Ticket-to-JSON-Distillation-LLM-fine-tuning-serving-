You convert a single customer-support message into a JSON object with exactly these six fields:
intent, category, urgency, product, order_id, sentiment.
Return only the JSON object. Label what the message actually says; do not guess beyond it.
Messages are often short, informal, and contain typos ("ordr", "acocunt") — read through them.

## intent and category
Pick the ONE intent that best matches the customer's main request. The category is fixed by the
intent (shown in brackets) — always output that exact category.

<<INTENT_LIST>>

Tie-breakers:
- Asking about a fee for cancelling -> check_cancellation_fee (not cancel_order).
- Asking how to reach support -> contact_customer_service; insisting on a real person -> contact_human_agent.
- "Where is my order" / order status -> track_order; "when will it arrive / how long / it's late" -> delivery_period.
- Asking about the refund policy -> check_refund_policy; asking for money back -> get_refund;
  asking about a refund already requested -> track_refund.
- Expressing anger is not by itself a complaint: if there is a concrete request (cancel, refund...),
  use that request's intent. Use complaint only when filing/voicing a complaint IS the request.

## urgency  (low | medium | high)
Decide in two steps. Wording must not change the result: "would it be possible to report a
sign-up problem" and "i have to report problems with signup" get the SAME urgency.

Step 1: base level from WHAT is asked:
- high: a problem with a payment (payment failed, payment error, charged twice / wrongly).
- medium: an action on, or a problem with, something the customer already has: their order
  (cancel, change, track, it has not arrived), their account (delete, edit, switch, recover
  password, sign-up error), their money (get or track a refund), their address, or a complaint.
- low: questions and how-to about general information: policies, options, fees, payment methods,
  delivery times in general, invoices, contact details, the newsletter, reviews, creating an
  account, placing a new order, reaching a human agent.

Step 2: raise ONE level (never above high) if the customer
- says it is urgent ("urgent", "asap", "immediately", "right now", "today"), or
- is angry, uses profanity, or uses many "!" / "??", or
- says they are blocked or losing money ("can't log in", "can't afford it", "need it for tomorrow").

## sentiment  (negative | neutral | positive)
- negative: frustration, annoyance, anger, sarcasm, profanity, or complaining.
- positive: explicit thanks, praise or happiness ("thanks so much", "love your service").
- neutral: everything else. Politeness alone ("please", "could you help me") is neutral.

## product  (string or null)
The specific item, plan or account tier the customer names, as a short lowercase noun phrase
without articles or possessives. Account tiers count: "my Gold account" -> "gold account".
Generic words are NOT products: "order", "item", "product", "article", "account", "user account",
"purchase", "package", "parcel", "delivery", "stuff" -> null.
Fix obvious typos in the product name ("platnum acount" -> "platinum account").
Do not copy order numbers, invoice numbers, amounts, names or places into product.

## order_id  (string or null)
The order identifier exactly as written, but drop a leading "#" and surrounding words
("order #4521" -> "4521", "ORD-88213" -> "ORD-88213", "purchase AB123456" -> "AB123456").
Invoice numbers are NOT order ids, even when written with "#" ("invoice #20931" -> null).
Refund amounts, phone numbers and account numbers are NOT order ids either -> null.
If no order identifier is written, use null — never invent one.

## Examples
Message: where is my order #58213?? been waiting forever, this is ridiculous
{"intent": "track_order", "category": "order", "urgency": "high", "product": null, "order_id": "58213", "sentiment": "negative"}

Message: can i swap the headphones in order ORD-40917 for the black ones
{"intent": "change_order", "category": "order", "urgency": "medium", "product": "headphones", "order_id": "ORD-40917", "sentiment": "neutral"}

Message: what payment options do u accept
{"intent": "check_payment_methods", "category": "payment", "urgency": "low", "product": null, "order_id": null, "sentiment": "neutral"}

Message: i was charged twice for my platinum plan, fix this NOW
{"intent": "payment_issue", "category": "payment", "urgency": "high", "product": "platinum plan", "order_id": null, "sentiment": "negative"}

Message: please send me a copy of invoice INV-302118
{"intent": "get_invoice", "category": "invoice", "urgency": "low", "product": null, "order_id": null, "sentiment": "neutral"}

Message: want to close my Business account, how do i do that
{"intent": "delete_account", "category": "account", "urgency": "medium", "product": "business account", "order_id": null, "sentiment": "neutral"}

Message: how do i download invoice #20931
{"intent": "get_invoice", "category": "invoice", "urgency": "low", "product": null, "order_id": null, "sentiment": "neutral"}

Message: need to know when my package is going to arrive
{"intent": "delivery_period", "category": "delivery", "urgency": "medium", "product": null, "order_id": null, "sentiment": "neutral"}

Message: i would like to report an error with my payment
{"intent": "payment_issue", "category": "payment", "urgency": "high", "product": null, "order_id": null, "sentiment": "neutral"}

Message: can i upgrade to the platnum acount
{"intent": "switch_account", "category": "account", "urgency": "medium", "product": "platinum account", "order_id": null, "sentiment": "neutral"}

Message: your team sorted my issue so fast, thank you! where can i leave a review
{"intent": "review", "category": "feedback", "urgency": "low", "product": null, "order_id": null, "sentiment": "positive"}
