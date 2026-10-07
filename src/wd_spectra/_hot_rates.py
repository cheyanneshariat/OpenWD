"""Exact fixed-material helium rate reuse for coupled-Newton experiments."""
import numpy as np
from . import helium_nlte as he
from .hot_nlte import HotPopulationState
from .constants import LIGHT_SPEED, PLANCK
from . import multilevel_nlte as hydrogen

_RADIATION_RESPONSE_WORKING_BYTES = 128 * 1024**2


def _continuum_response_integrals(mean_response, boltzmann, weights):
    """Integrate the same rates without a second full radiation tangent.

    A 185-depth tangent occupies several gigabytes. Bounding the temporary
    stimulated-emission array avoids evicting the reusable transfer bases.
    Chunking changes only the order of the wavelength summation.
    """
    nw, nd, count = mean_response.shape
    block = max(1, _RADIATION_RESPONSE_WORKING_BYTES // (nd*count*8))
    if block >= nw:
        plain = mean_response.transpose(1,2,0).reshape(nd*count,-1)
        stimulated = (mean_response*boltzmann[:,:,None]).transpose(1,2,0).reshape(nd*count,-1)
        up, down = plain@weights, stimulated@weights
    else:
        up = np.zeros((nd*count,weights.shape[1]))
        down = np.zeros_like(up)
        for start in range(0,nw,block):
            stop = min(start+block,nw)
            local = mean_response[start:stop]
            up += local.reshape(stop-start,-1).T@weights[start:stop]
            stimulated = local*boltzmann[start:stop,:,None]
            down += stimulated.reshape(stop-start,-1).T@weights[start:stop]
    return (up.reshape(nd,count,-1).transpose(0,2,1),
            down.reshape(nd,count,-1).transpose(0,2,1))


class PreparedHeliumRates:
    def __init__(self, model, atmosphere, wave, groups):
        self.model,self.atmosphere,self.wave=model,atmosphere,wave
        self.kwargs=dict(maximum_helium_ii_level=model.maximum_helium_ii_level,
            helium_i_collision_data=model.helium_i_collision_data,
            hydrogenic_collision_model=model.hydrogenic_collision_model)
        self.neutral,self.total_ion,self.neutral_occupation=he._neutral_helium_reference_populations(atmosphere)
        self.ion,self.continuum,self.ion_occupation=he._reference_populations(atmosphere,model.maximum_helium_ii_level)
        zero=np.zeros(atmosphere.n_depth);one=np.ones(atmosphere.n_depth)
        zero_mean=np.zeros((len(wave),atmosphere.n_depth))
        self.base=he.solve_coupled_helium_statistical_equilibrium(atmosphere,model.collision_data,
            **self.kwargs,neutral_line_mean_intensity_nu={k:zero for k in groups[0]},
            helium_ii_line_mean_intensity_nu={k:zero for k in groups[1]},
            neutral_continuum_wavelength_angstrom=wave,neutral_continuum_mean_intensity_lambda=zero_mean,
            helium_ii_continuum_wavelength_angstrom=wave,helium_ii_continuum_mean_intensity_lambda=zero_mean,
            _return_rate_matrix=True)
        self.line_updates=[]
        for group_number,group in enumerate(groups[:2]):
            keys=[];lower_indices=[];upper_indices=[];upward=[];downward=[]
            for lower,upper in group:
                if group_number==0:
                    f=he.HELIUM_I_14_OSCILLATOR_STRENGTH.get((lower,upper),0.)
                    if f<=0:continue
                    nu=he.HELIUM_I_14_THRESHOLD_FREQUENCY_HZ[lower-1]-he.HELIUM_I_14_THRESHOLD_FREQUENCY_HZ[upper-1]
                    up,_=he._neutral_helium_bound_bound_radiative_rates(atmosphere,lower-1,upper-1,f,self.neutral,self.neutral_occupation,one)
                    _,spont=he._neutral_helium_bound_bound_radiative_rates(atmosphere,lower-1,upper-1,f,self.neutral,self.neutral_occupation,zero)
                    lo,hi=lower-1,upper-1
                else:
                    line=he.helium_ii_shell_transition(lower,upper)
                    nu=LIGHT_SPEED*1e8/line.wavelength_vacuum_angstrom
                    args=(atmosphere.temperature,line,self.ion[:,lower-1],self.ion[:,upper-1],self.ion_occupation[:,lower-1],self.ion_occupation[:,upper-1])
                    up,_=he._bound_bound_radiative_rates(*args,one)
                    _,spont=he._bound_bound_radiative_rates(*args,zero)
                    lo,hi=14+lower-1,14+upper-1
                keys.append((lower,upper));lower_indices.append(lo);upper_indices.append(hi)
                upward.append(up);downward.append(spont*LIGHT_SPEED**2/(2*PLANCK*nu**3))
            self.line_updates.append((keys,np.array(lower_indices),np.array(upper_indices),np.array(upward).T,np.array(downward).T))
        self.neutral_zero=he._neutral_helium_continuum_radiative_rates(atmosphere,self.neutral,self.total_ion,wave,zero_mean)[1]
        self.ion_zero=he._continuum_radiative_rates(atmosphere,self.ion,self.continuum,wave,zero_mean)[1]
        self.ground_fraction=self.ion[:,0]/np.maximum(self.total_ion,np.finfo(float).tiny)

    def rate_matrix(self, mean, neutral_fields, ion_fields):
        rate=self.base.copy()
        for fields,(keys,lo,hi,up,down) in zip((neutral_fields,ion_fields),self.line_updates):
            intensity=np.column_stack([fields[k] for k in keys])
            rate[:,lo,hi]+=up*intensity
            rate[:,hi,lo]+=down*intensity
        up,down=he._neutral_helium_continuum_radiative_rates(self.atmosphere,self.neutral,self.total_ion,self.wave,mean)
        rate[:,:14,14]+=up
        rate[:,14,:14]+=(down-self.neutral_zero)/np.maximum(self.ground_fraction[:,None],np.finfo(float).tiny)
        up,down=he._continuum_radiative_rates(self.atmosphere,self.ion,self.continuum,self.wave,mean)
        rate[:,14:-1,-1]+=up
        rate[:,-1,14:-1]+=down-self.ion_zero
        return rate

    def state(self, mean, neutral_fields, ion_fields, hydrogen_fields):
        rate = self.rate_matrix(mean, neutral_fields, ion_fields)
        helium=he.solve_coupled_helium_statistical_equilibrium(self.atmosphere,self.model.collision_data,
            **self.kwargs,neutral_line_mean_intensity_nu=neutral_fields,helium_ii_line_mean_intensity_nu=ion_fields,
            _rate_matrix=rate)
        h=None
        if self.model.log_hydrogen_to_helium is not None:
            h=hydrogen.solve_multilevel_hydrogen_statistical_equilibrium(self.atmosphere,self.model.collision_data,
                maximum_level=self.model.maximum_hydrogen_level,line_mean_intensity_nu=hydrogen_fields,
                continuum_wavelength_angstrom=self.wave,continuum_mean_intensity_lambda=mean)
        state = HotPopulationState(helium,h)
        # The caller may attach these to a completed radiation evaluation.
        # They are never reused by matching only an atmosphere or array id.
        self.last_rate_matrix = rate
        return state


class PreparedLineAverages:
    """Sparse form of the unchanged per-depth profile/oscillator quadrature."""
    def __init__(self,wave,groups,depths):
        from scipy.sparse import coo_matrix
        self.groups=groups
        self.keys=[list(group) for group in groups]
        rows=[];columns=[];values=[];line_index=0
        for group in groups:
            for problems in group.values():
                total=sum(p.line.absorption_oscillator_strength for p in problems)
                for p in problems:
                    wavelength=p.continuum.wavelength_angstrom
                    spacing=np.diff(wavelength)
                    quadrature=.5*(np.r_[0.,spacing]+np.r_[spacing,0.])
                    weighted=p.lte_line_opacity*quadrature[:,None]
                    denominator=(weighted*(LIGHT_SPEED*1e8/wavelength[:,None]**2)).sum(axis=0)
                    kernel=np.divide(weighted,denominator,out=np.zeros_like(weighted),where=denominator>0)
                    kernel*=p.line.absorption_oscillator_strength/total
                    index=np.searchsorted(wave,wavelength)
                    np.testing.assert_array_equal(wave[index],wavelength)
                    columns.append((index[:,None]*depths+np.arange(depths)).ravel())
                    rows.append(np.broadcast_to(line_index*depths+np.arange(depths),kernel.shape).ravel())
                    values.append(kernel.ravel())
                line_index+=1
        self.matrix=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
            shape=(line_index*depths,len(wave)*depths)).tocsr()
        self.depths=depths

    def fields(self,mean):
        values=(self.matrix@mean.ravel()).reshape(-1,self.depths)
        offset=0;groups=[]
        for keys in self.keys:
            groups.append({key:values[offset+i] for i,key in enumerate(keys)})
            offset+=len(keys)
        return groups


class HeliumRadiationResponse:
    """Differentiate the normalized SE solve for many radiation directions.

    At fixed material state all radiative rates are affine in J. Reuse one
    factorization per depth and solve its differentiated conservation system;
    no radiation/population lag or approximate lambda operator is introduced.
    """
    def __init__(self, rates, profiles, mean, *, state=None, rate_matrix=None):
        # Supplied state/rates must belong to this exact material and mean field.
        from scipy.linalg import lu_factor
        from .hot_nlte import population_arrays
        from ._nlte_radiative_integrals import _kernel, _boltzmann
        self.rates,self.profiles=rates,profiles
        fields=profiles.fields(mean)
        if rate_matrix is None:
            rate_matrix = rates.rate_matrix(mean, *fields[:2])
        if state is None:
            # A helium-only response must not solve and discard hydrogen.
            helium = he.solve_coupled_helium_statistical_equilibrium(
                rates.atmosphere, rates.model.collision_data, **rates.kwargs,
                neutral_line_mean_intensity_nu=fields[0],
                helium_ii_line_mean_intensity_nu=fields[1], _rate_matrix=rate_matrix)
            state = HotPopulationState(helium)
        population,reference=population_arrays(state)
        # Hydrogen has its own conservation row and response below.
        n=15+rates.model.maximum_helium_ii_level
        population,reference=population[:,:n],reference[:,:n]
        density=reference.sum(axis=1)[:,None]
        self.reference=reference/density
        self.population=population/density
        self.departure=population/reference
        rate=rate_matrix
        matrix=rate.transpose(0,2,1).copy()
        diagonal=np.arange(n)
        matrix[:,diagonal,diagonal]-=rate.sum(axis=2)
        matrix*=self.reference[:,None,:]
        matrix[:,-1,:]=self.reference
        self.scale=np.maximum(abs(matrix).max(axis=2),np.finfo(float).tiny)
        self.factors=[lu_factor(m/s[:,None]) for m,s in zip(matrix,self.scale)]
        key=np.asarray(rates.wave,dtype=np.float64).tobytes()
        self.boltzmann=_boltzmann(key,np.asarray(rates.atmosphere.temperature,dtype=np.float64).tobytes())
        self.continua=[]
        for count,first,cross,lower,upper in (
            (14,0,he.neutral_helium_term_photoionization_cross_section,np.arange(14),14),
            (rates.model.maximum_helium_ii_level,1,he.helium_ii_photoionization_cross_section,np.arange(14,n-1),n-1)):
            weights,_=_kernel(key,count,first,cross)
            self.continua.append((weights,lower,upper))

    def log_ratio_response(self, mean_response):
        from scipy.linalg import lu_solve
        nd,nstate=self.population.shape
        count=mean_response.shape[2]
        rhs=np.zeros((nd,nstate,count))
        line_response=(self.profiles.matrix@mean_response.reshape(-1,count)).reshape(-1,nd,count)
        offset=0
        for keys,updates in zip(self.profiles.keys[:2],self.rates.line_updates):
            update_keys,lo,hi,up,down=updates
            lookup={key:offset+i for i,key in enumerate(keys)}
            for i,key in enumerate(update_keys):
                net=(self.population[:,lo[i]]*up[:,i]-self.population[:,hi[i]]*down[:,i])[:,None]*line_response[lookup[key]]
                rhs[:,lo[i],:]+=net
                rhs[:,hi[i],:]-=net
            offset+=len(keys)
        for weights,lower,upper in self.continua:
            up,down=_continuum_response_integrals(mean_response,self.boltzmann,weights)
            down*=self.reference[:,lower,None]/self.reference[:,upper,None,None]
            net=self.population[:,lower,None]*up-self.population[:,upper,None,None]*down
            rhs[:,lower,:]+=net
            rhs[:,upper,:]-=net.sum(axis=1)
        rhs[:,-1,:]=0.
        db=np.array([lu_solve(f,r/s[:,None]) for f,r,s in zip(self.factors,rhs,self.scale)])
        # Match the original solver's final explicit conservation assignment.
        db[:,-1,:]=-np.sum(db[:,:-1,:]*self.reference[:,:-1,None],axis=1)/self.reference[:,-1,None]
        dlog=db/self.departure[:,:,None]
        return dlog[:,:-1,:]-dlog[:,-1:, :]


class HydrogenRadiationResponse:
    """Exact fixed-material H response, with the atom's conservation closure."""
    def __init__(self, rates, profiles, mean, *, state=None):
        from scipy.linalg import lu_factor
        from ._nlte_radiative_integrals import _kernel, _boltzmann
        a,model=rates.atmosphere,rates.model
        self.profiles=profiles
        levels=model.maximum_hydrogen_level
        fields=profiles.fields(mean)[2]
        arguments=dict(maximum_level=levels,line_mean_intensity_nu=fields,
                       continuum_wavelength_angstrom=rates.wave,continuum_mean_intensity_lambda=mean)
        if state is None:
            state=hydrogen.solve_multilevel_hydrogen_statistical_equilibrium(a,model.collision_data,**arguments)
        rate=hydrogen.solve_multilevel_hydrogen_statistical_equilibrium(a,model.collision_data,
                    **arguments,_return_rate_matrix=True)
        reference=np.column_stack((state.lte_population_density,state.lte_proton_density))
        population=np.column_stack((state.population_density,state.proton_density))
        density=reference.sum(axis=1)[:,None]
        self.reference,self.population=reference/density,population/density
        self.departure=population/reference
        self.inactive=self.reference[:,:-1]<1e-60
        matrix=rate.transpose(0,2,1).copy()
        diagonal=np.arange(levels+1)
        matrix[:,diagonal,diagonal]-=rate.sum(axis=2)
        matrix*=self.reference[:,None,:]
        matrix[:,-1,:]=self.reference
        self.scale=np.maximum(abs(matrix).max(axis=2),np.finfo(float).tiny)
        self.factors=[lu_factor(m/s[:,None]) for m,s in zip(matrix,self.scale)]
        self.lines=[]
        occupation=hydrogen._reference_populations(a,levels)[2]
        zero=np.zeros(a.n_depth);one=np.ones(a.n_depth)
        for lower,upper in profiles.keys[2]:
            line=(hydrogen.hydrogen_shell_transition(lower,upper) if upper <= 9 else
                  hydrogen._extended_hydrogen_shell_transition(lower,upper))
            args=(a.temperature,line,reference[:,lower-1],reference[:,upper-1],
                  occupation[:,lower-1],occupation[:,upper-1])
            up,_=hydrogen._bound_bound_radiative_rates(*args,one)
            _,spontaneous=hydrogen._bound_bound_radiative_rates(*args,zero)
            frequency=LIGHT_SPEED*1e8/line.wavelength_vacuum_angstrom
            down=spontaneous*LIGHT_SPEED**2/(2*PLANCK*frequency**3)
            self.lines.append((lower-1,upper-1,up,down))
        key=np.asarray(rates.wave,dtype=np.float64).tobytes()
        self.weights,_=_kernel(key,levels,1,hydrogen._photoionization_cross_section)
        self.boltzmann=_boltzmann(key,np.asarray(a.temperature,dtype=np.float64).tobytes())

    def log_ratio_response(self, mean_response):
        from scipy.linalg import lu_solve
        nd,nstate=self.population.shape
        count=mean_response.shape[2]
        rhs=np.zeros((nd,nstate,count))
        lines=(self.profiles.matrix@mean_response.reshape(-1,count)).reshape(-1,nd,count)
        offset=sum(map(len,self.profiles.keys[:2]))
        for i,(lo,hi,up,down) in enumerate(self.lines):
            net=(self.population[:,lo]*up-self.population[:,hi]*down)[:,None]*lines[offset+i]
            rhs[:,lo,:]+=net
            rhs[:,hi,:]-=net
        up,down=_continuum_response_integrals(mean_response,self.boltzmann,self.weights)
        down*=self.reference[:,:-1,None]/self.reference[:,-1,None,None]
        rhs[:,:-1,:]+=self.population[:,:-1,None]*up-self.population[:,-1,None,None]*down
        rhs[:,-1,:]=0.
        db=np.array([lu_solve(f,r/s[:,None]) for f,r,s in zip(self.factors,rhs,self.scale)])
        db[:,:-1,:][self.inactive]=0.
        db[:,-1,:]=-np.sum(db[:,:-1,:]*self.reference[:,:-1,None],axis=1)/self.reference[:,-1,None]
        dlog=db/self.departure[:,:,None]
        return dlog[:,:-1,:]-dlog[:,-1:,:]


class MixedRadiationResponse:
    """Concatenate independently conserved H and He responses to the same J."""
    def __init__(self,rates,profiles,mean,*,state=None,rate_matrix=None):
        self.helium=HeliumRadiationResponse(rates,profiles,mean,state=state,rate_matrix=rate_matrix)
        self.hydrogen=HydrogenRadiationResponse(rates,profiles,mean,
                                               state=None if state is None else state.hydrogen)

    def log_ratio_response(self,mean_response):
        return np.concatenate((self.helium.log_ratio_response(mean_response),
                               self.hydrogen.log_ratio_response(mean_response)),axis=1)
