# Problem Statement

*Agentic AI Hackathon 2026 — "The AgentIQ Frontier"*

> This is a reworded summary of the challenge, reconstructed from the project's
> design documents. It is not the organisers' original text.

## Context

**Urban Media** is a fictional Digital-Out-Of-Home (DOOH) advertising company.
It sells ad space on an inventory of **11,163 screens** (transit and street
displays) spread across three markets:

| Market | Positioning |
|---|---|
| Las Hackland | Premium |
| Accordionshire | Value |
| DA Town | Standard |

Today a salesperson receives a client brief (a Word document), then decides by
hand which screens to book, in which time blocks, and at what price. This is
slow, inconsistent between reps, and hard to defend or learn from.

## The challenge

Build an AI-assisted commercial intelligence platform that turns a client
brief into a media plan a salesperson can send, and explains the reasoning
behind it. The system should combine machine learning, optimisation and an LLM
agent, each used where it is the right tool.

The challenge was split into six deliverables:

| # | Deliverable | Goal |
|---|---|---|
| D1 | Audience Profile Engine | Build deduplicated audience pools for each screen and time block from footfall, transit and points-of-interest data |
| D2 | Campaign–Screen Relevance Scorer | Score how well each candidate screen fits a campaign, with reasons |
| D3 | Demand Forecasting & Pricing | Recommend a price band (floor / target / cap) and a win probability for each line |
| D4 | Impressions Optimizer | Choose the set of screens and slots that maximises impressions under budget and capacity limits |
| D5 | Agentic Orchestration | Coordinate the engines, apply rep feedback, and explain the final plan, including what it could not do |
| D6 | Unified Platform | A single application a sales rep can use end to end |

## Key requirements and difficulties

- **Non-linear impressions.** Ads that play more slots per minute are more
  likely to be noticed, so impressions do not scale linearly with boards or
  spend. The optimiser has to account for this.
- **Sparse and censored pricing data.** Win/loss history only shows prices that
  were accepted or rejected, and many screen–slot combinations have no
  booking history at all. Pricing has to handle cold starts honestly.
- **Constrained decisions belong to optimisation, not the LLM.** Choosing
  screens under a budget and capacity is an integer-programming problem; the
  LLM is used to interpret briefs and narrate results.
- **Be honest about limits.** The plan should state what the system could not
  do or is unsure about, rather than hide it.
- **Cost control.** The event provided a small shared LLM budget, so the
  platform needed spend guards and deterministic fallbacks.

## What success looks like

For a given brief, a rep gets a ranked screen shortlist, a priced and
optimised package, the reach and frequency it buys, and a written
recommendation they can send to the client.

## This repository's approach

See [README.md](README.md) for the architecture, [D6_APP_DESIGN.md](D6_APP_DESIGN.md)
for the application design, and [C4_SYSTEM_DESIGN.html](C4_SYSTEM_DESIGN.html)
for the system design.
