import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { User, Session } from "@supabase/supabase-js";

export type StaffRole = "admin" | "attending" | "nurse";

export interface UserProfile {
  id: string;
  user_id: string;
  full_name: string | null;
  role: StaffRole;
  department: string | null;
  employee_id: string | null;
  hospital_id: string | null;
  avatar_url: string | null;
  hospital_name: string | null;
}

interface AuthContextType {
  user: User | null;
  session: Session | null;
  profile: UserProfile | null;
  loading: boolean;
  signOut: () => Promise<void>;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  profile: null,
  loading: true,
  signOut: async () => {},
  refreshProfile: async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchProfile = async (userId: string) => {
    const { data, error } = await supabase
      .from("profiles")
      .select("*")
      .eq("user_id", userId)
      .maybeSingle();

    if (error || !data) {
      setProfile(null);
      return;
    }

    // Fetch hospital name if hospital_id exists
    let hospitalName: string | null = null;
    if (data.hospital_id) {
      const { data: hospital } = await supabase
        .from("hospitals")
        .select("name")
        .eq("id", data.hospital_id)
        .maybeSingle();
      hospitalName = hospital?.name ?? null;
    }

    setProfile({
      id: data.id,
      user_id: data.user_id,
      full_name: data.full_name,
      role: (data.role as StaffRole) || "nurse",
      department: data.department,
      employee_id: data.employee_id,
      hospital_id: data.hospital_id,
      avatar_url: data.avatar_url,
      hospital_name: hospitalName,
    });
  };

  const refreshProfile = async () => {
    if (user) await fetchProfile(user.id);
  };

  useEffect(() => {
    // IMPORTANT: Set up listener BEFORE getSession to avoid missing events
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        if (session?.user) {
          // Fire and forget - never await inside onAuthStateChange
          fetchProfile(session.user.id).then(() => setLoading(false));
        } else {
          setProfile(null);
          setLoading(false);
        }
      }
    );

    // Restore session from storage
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        fetchProfile(session.user.id).then(() => setLoading(false));
      } else {
        setLoading(false);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const signOut = async () => {
    await supabase.auth.signOut();
    setProfile(null);
  };

  return (
    <AuthContext.Provider value={{ user, session, profile, loading, signOut, refreshProfile }}>
      {children}
    </AuthContext.Provider>
  );
}
