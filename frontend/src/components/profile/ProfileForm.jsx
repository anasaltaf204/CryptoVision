/**
 * components/profile/ProfileForm.jsx
 *
 * Reusable controlled form for editing user profile fields.
 * Owns no async logic — calls onSubmit(data) and receives
 * saveStatus/saveError from the parent (ProfilePage).
 *
 * This separation makes the form independently testable and reusable
 * (e.g. an inline edit widget in a future settings page).
 */

import React, { useState, useEffect } from "react";

export default function ProfileForm({
    initialValues = {},
    onSubmit,
    saveStatus,  // "idle" | "saving" | "saved" | "error"
    saveError,
}) {
    const [values, setValues] = useState({
        username: "",
        email:    "",
        bio:      "",
        ...initialValues,
    });
    const [touched, setTouched] = useState({});

    // Sync form if initialValues change (e.g. after Firestore loads)
    useEffect(() => {
        setValues((prev) => ({ ...prev, ...initialValues }));
    }, [initialValues.username, initialValues.email, initialValues.bio]);

    function handleChange(e) {
        const { name, value } = e.target;
        setValues((prev) => ({ ...prev, [name]: value }));
        setTouched((prev) => ({ ...prev, [name]: true }));
    }

    function handleSubmit(e) {
        e.preventDefault();
        // Only send fields the user actually touched
        const changes = Object.fromEntries(
            Object.entries(values).filter(([k]) => touched[k])
        );
        if (Object.keys(changes).length === 0) return;
        onSubmit(changes);
    }

    const isSaving  = saveStatus === "saving";
    const isSaved   = saveStatus === "saved";
    const isError   = saveStatus === "error";

    return (
        <form className="profile-form" onSubmit={handleSubmit} noValidate>
            <div className="profile-form__field">
                <label htmlFor="pf-username" className="profile-form__label">
                    Display Name
                </label>
                <input
                    id="pf-username"
                    name="username"
                    type="text"
                    className="profile-form__input"
                    value={values.username}
                    onChange={handleChange}
                    placeholder="Your display name"
                    maxLength={40}
                    required
                />
            </div>

            <div className="profile-form__field">
                <label htmlFor="pf-email" className="profile-form__label">
                    Email
                </label>
                <input
                    id="pf-email"
                    name="email"
                    type="email"
                    className="profile-form__input profile-form__input--readonly"
                    value={values.email}
                    readOnly
                    tabIndex={-1}
                    title="Email cannot be changed"
                />
                <p className="profile-form__hint">Email address cannot be changed.</p>
            </div>

            <div className="profile-form__field">
                <label htmlFor="pf-bio" className="profile-form__label">
                    Bio <span className="profile-form__optional">(optional)</span>
                </label>
                <textarea
                    id="pf-bio"
                    name="bio"
                    className="profile-form__input profile-form__textarea"
                    value={values.bio}
                    onChange={handleChange}
                    placeholder="Tell us a bit about yourself…"
                    rows={3}
                    maxLength={200}
                />
                <p className="profile-form__char-count">
                    {values.bio?.length || 0} / 200
                </p>
            </div>

            {/* Save feedback */}
            {isSaved && (
                <div className="profile-form__banner profile-form__banner--success">
                    ✓ Profile saved successfully
                </div>
            )}
            {isError && (
                <div className="profile-form__banner profile-form__banner--error">
                    ✗ {saveError || "Failed to save profile. Try again."}
                </div>
            )}

            <button
                type="submit"
                className="profile-form__submit"
                disabled={isSaving || Object.keys(touched).length === 0}
            >
                {isSaving ? (
                    <>
                        <span className="profile-form__spinner" />
                        Saving…
                    </>
                ) : "Save Changes"}
            </button>
        </form>
    );
}
