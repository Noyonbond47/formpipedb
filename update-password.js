const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';

const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const updatePasswordForm = document.getElementById('update-password-form');
const messageDiv = document.getElementById('message');

updatePasswordForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const newPassword = event.target.newPassword.value;
    messageDiv.textContent = 'Updating...';

    // Use the updateUser method to set the new password.
    // This works because the user has a valid, temporary session
    // from clicking the recovery link.
    const { data, error } = await supabase.auth.updateUser({
        password: newPassword
    });

    if (error) {
        messageDiv.textContent = `Error: ${error.message}`;
        messageDiv.style.color = 'red';
    } else {
        messageDiv.textContent = 'Password updated successfully! You can now log in.';
        messageDiv.style.color = 'green';
        // Optionally, redirect to the login page after a short delay
        // setTimeout(() => { window.location.href = '/login.html'; }, 3000);
    }
});