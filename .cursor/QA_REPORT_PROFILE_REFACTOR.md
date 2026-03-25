# Profile Refactoring — QA Report

## Executive Summary

Static code analysis and UX/UI audit of profile components (`frontend/src/pages/profile/`) and `styles.css` identified **8 bugs** and **3 recommendations**. Below are findings with exact fix snippets.

---

## 1. Mobile Responsiveness & Layout Shifts

### BUG 1.1: Inline grid styles override mobile media queries

**Location:** `SubscriptionTab.jsx`, `EmployerDashboard.jsx`

**Issue:** Inline `style={{ gridTemplateColumns: "..." }}` overrides CSS media queries. On viewports < 480px, `subscription-tiers-grid` should collapse to 1 column, but the inline style wins.

**Fix for SubscriptionTab.jsx** — Remove inline style, rely on CSS:

```jsx
// BEFORE (line ~235):
<div
  className="subscription-tiers-grid"
  style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}
>

// AFTER:
<div className="subscription-tiers-grid">
```

CSS already provides: `.subscription-tiers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; }` and `@media (max-width: 480px) { .subscription-tiers-grid { grid-template-columns: 1fr; gap: 1rem; } }`.

---

### BUG 1.2: EmployerDashboard metrics grid ignores mobile optimizations

**Location:** `EmployerDashboard.jsx` lines 201–209

**Issue:** Inline style overrides `.dashboard-metrics` media query, so at 768px the smaller padding/gap and 2-column layout are lost.

**Fix for EmployerDashboard.jsx** — Remove inline style:

```jsx
// BEFORE:
<div
  className="dashboard-metrics"
  style={{
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "1.5rem",
    marginBottom: "2rem",
  }}
>

// AFTER:
<div className="dashboard-metrics" style={{ marginBottom: "2rem" }}>
```

Add to `styles.css` if `marginBottom` is desired at all breakpoints:

```css
.dashboard-section--metrics .dashboard-metrics {
  margin-bottom: 2rem;
}
```

---

### BUG 1.3: Long text can break layout in profile cards

**Location:** `styles.css` — `.profile-list-title`, `.profile-list-content`

**Issue:** Long names, emails, or titles can overflow and break layout. No `word-break` or `overflow` handling.

**Fix in styles.css** — Add after `.profile-list-title-link:hover` (~line 7082):

```css
.profile-list-title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  min-width: 0; /* allow shrinking */
}

.profile-list-title > span,
.profile-list-title > a,
.profile-list-title-link {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.profile-list-text {
  word-break: break-word;
  overflow-wrap: break-word;
}
```

Note: `.profile-list-title` already exists. Merge with existing rule:

```css
/* Add to existing .profile-list-title block */
.profile-list-title {
  font-weight: 600;
  margin-bottom: 0.2rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}

.profile-list-title > span,
.profile-list-title .profile-list-title-link {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  flex: 1 1 auto;
}
```

---

### BUG 1.4: Primary action buttons may shrink in profile-section-header

**Location:** `styles.css` — `.profile-section-header`

**Issue:** On narrow screens, the Add button can shrink. Primary actions should have `flex-shrink: 0`.

**Fix in styles.css** — Add:

```css
.profile-section-header > .primary-btn,
.profile-section-header > .secondary-btn,
.profile-section-header .profile-section-card__title + *,
.profile-section-header .profile-section-card__title {
  flex-shrink: 0;
}
```

Or more targeted:

```css
.profile-section-header .primary-btn,
.profile-section-header .secondary-btn,
.profile-section-header .ghost-btn {
  flex-shrink: 0;
}
```

---

## 2. Card-in-Card Artifacts

### BUG 2.1: SubscriptionTab status card — Card vs legacy class conflict

**Location:** `SubscriptionTab.jsx` line 168

**Issue:** Status block uses both `<Card variant="elevated">` and `.subscription-status-card` which has its own background/border. This can cause double borders or visual clash.

**Current:** `className={active ? "subscription-status-card subscription-status-card--active" : "..."}` — `.subscription-status-card` in CSS sets `background`, `border`, etc., which may clash with `ui-card--elevated`.

**Fix:** Either (a) remove Card and use a styled div, or (b) remove `.subscription-status-card` custom styling when used with Card. Simplest: keep Card, ensure `.subscription-status-card` only adds layout (flex), not background/border:

```css
/* In styles.css — ensure subscription-status-card doesn't add conflicting styles when inside Card */
.subscription-status-card.ui-card {
  /* override subscription-status-card background if it conflicts */
  background: transparent; /* let ui-card handle it */
  border: none; /* let ui-card handle it */
}
```

Or: revert status block to a `<div>` with the original classes and avoid Card there.

---

## 3. Input Validation & Form Stability

### BUG 3.1: Input error causes Cumulative Layout Shift (CLS)

**Location:** `styles.css` — `.ui-input-group`, `.ui-input-error`

**Issue:** When `error` appears, the error message increases height and shifts content below.

**Fix in styles.css** — Reserve space or use absolute positioning:

```css
.ui-input-group {
  position: relative;
  min-height: 2.5rem; /* reserve space for input */
}

.ui-input-error {
  position: absolute;
  top: 100%;
  left: 0;
  margin-top: 0.2rem;
  font-size: 0.8125rem;
  color: var(--danger);
}
```

Or reserve a fixed minimum height for the error row:

```css
.ui-input-group {
  min-height: calc(2.5rem + 1.5rem); /* input height + one line for error */
}
```

---

### BUG 3.2: profile-form__row alignment when one input has error

**Location:** `styles.css` — `.profile-form__row`

**Issue:** If one input has an error and the other does not, `align-items: stretch` (default) makes both cells equal height, which is acceptable. To prevent visual misalignment when errors appear at different times, use:

```css
.profile-form__row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  align-items: start; /* top-align so error in one cell doesn't push the other */
}
```

---

## 4. Sticky Elements & Z-Index

### BUG 4.1: Profile sidebar missing z-index

**Location:** `styles.css` — `.profile-page__sidebar`

**Issue:** `position: sticky` without `z-index` can lead to incorrect stacking vs dropdowns (e.g. GlobalSearch z-index 10000) or modals.

**Fix in styles.css** — Add after `.profile-page__sidebar` block:

```css
.profile-page__sidebar {
  position: sticky;
  top: 5rem;
  z-index: 10; /* below header (500), below dropdowns (10000), but above content */
  /* ... rest unchanged */
}
```

---

## 5. Borders and Dividers

### OK: dashboard-list-item__actions

**Finding:** `.dashboard-list-item__actions` uses `border-top: 1px solid var(--border-light)`. The parent `Card` has its own border. The top border is a divider between content and actions — no redundant border observed. Last child has no bottom border; layout is consistent.

**Optional:** If the divider feels too strong, reduce to:

```css
.dashboard-list-item__actions {
  border-top: 1px solid var(--border-light);
  /* consider: border-top-color: transparent; or a lighter color */
}
```

---

## 6. Routing and Transitions

### OK: stopPropagation usage

**Finding:** `MyJoinRequestsSection.jsx` RequestCard "Покинуть" correctly uses `e.stopPropagation()`. `MyVacancyResponsesSection.jsx` ResponseCard is a full Link with no nested buttons — no conflict. `VacancyResponsesIncomingTab` has Link only on applicant name; status select does not need stopPropagation. No double-routing issues found.

---

## 7. Additional Recommendations

### REC 7.1: lab-form-actions primary button full-width on mobile

**Location:** `styles.css`

**Suggestion:** On small screens, make Save/Create full width for touch targets:

```css
@media (max-width: 480px) {
  .lab-form-actions .primary-btn,
  .lab-form-actions--create .primary-btn {
    width: 100%;
  }
}
```

---

### REC 7.2: dashboard-list-item__actions button shrink

**Location:** `styles.css`

**Suggestion:** Prevent action buttons from shrinking:

```css
.dashboard-list-item__actions .primary-btn,
.dashboard-list-item__actions .ghost-btn,
.dashboard-list-item__actions button,
.dashboard-list-item__actions select {
  flex-shrink: 0;
}
```

---

### REC 7.3: subscription-tier-card on mobile (Card + class)

**Location:** `SubscriptionTab.jsx`

**Note:** Tier cards use `<Card variant="elevated">` with `className="subscription-tier-card"`. `.subscription-tier-card` in CSS has its own `background`, `border`, `padding`. The Card component adds `ui-card--elevated` and padding. This can cause double backgrounds. Consider either:

- Using only Card and removing `.subscription-tier-card` background/border, or
- Using only a `div` with `.subscription-tier-card` (and variants) without Card.

---

## Summary of Fixes to Apply

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1.1 | High | SubscriptionTab.jsx | Remove inline grid style |
| 1.2 | High | EmployerDashboard.jsx | Remove inline grid style (or add margin to CSS) |
| 1.3 | Medium | styles.css | Add overflow/word-break to profile-list-title, profile-list-text |
| 1.4 | Medium | styles.css | Add flex-shrink: 0 to profile-section-header buttons |
| 2.1 | Low | SubscriptionTab / styles | Resolve Card vs subscription-status-card style conflict |
| 3.1 | Medium | styles.css | Fix ui-input-error layout shift (reserve space or absolute) |
| 3.2 | Low | styles.css | Add align-items: start to profile-form__row |
| 4.1 | Medium | styles.css | Add z-index to profile-page__sidebar |
