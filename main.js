const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';

// Initialize the Supabase client
const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

console.log('Auth listener is active.');

// Listen for authentication state changes
supabase.auth.onAuthStateChange((event, session) => {
    console.log(`Auth event: ${event}`, session);

    // This is the key part for password recovery.
    // When the user clicks the link in the email, Supabase redirects them here,
    // and this event is triggered.
    if (event === 'PASSWORD_RECOVERY') {
        console.log('Password recovery event detected. Redirecting to update password page.');
        // Redirect the user to your dedicated "update password" page.
        window.location.href = '/frontend/update-password.html';
    }
});